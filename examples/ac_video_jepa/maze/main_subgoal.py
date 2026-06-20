"""Train the high-level SUBGOAL predictor (learned replacement for A* waypoints).

Feudal/closed-loop hierarchy: SubgoalPredictor(z_current, goal_xy) -> position of
the next waypoint ~N cells along the route to the goal. Supervised on A*
trajectories (label = the A* position N frames ahead). The fine world model +
encoder + probe are frozen. At eval (eval_subgoal.py) the predictor proposes the
waypoints and a low-level reacher follows them — NO A* at eval.

Run: python -m examples.ac_video_jepa.maze.main_subgoal <fine_ckpt> <out_dir> [N=4] [epochs=12]

    Smoke-test extras, kept optional so the default experiment is unchanged:
    [batch_size=96] [size_from_ckpt_config] [num_workers_from_ckpt_config]
    [block_penalty=0.0] [block_radius_px=1]
"""
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.optim import AdamW

from eb_jepa.datasets.utils import init_data
from eb_jepa.hierarchical import SubgoalPredictor
from eb_jepa.state_decoder import MLPXYHead
from eb_jepa.training_utils import load_checkpoint
from examples.ac_video_jepa.maze.maze_fine_wm import build_fine


def observed_obstacle_penalty(pred_norm, obs, normalizer, radius_px=1):
    """Penalty for predicted waypoints that land on visible obstacles.

    ``pred_norm`` is in the same normalized row/col coordinate system as the
    location labels. Dynamic-maze observations use channel 1 for visible walls,
    which also marks currently closed doors. Unknown fog is deliberately not
    penalized, otherwise the subgoal head learns the conservative policy that
    already fails in this benchmark.
    """
    if obs.shape[1] < 2:
        return pred_norm.new_tensor(0.0)

    state = normalizer.unnormalize_state(obs.float()).clamp(0.0, 1.0)
    obstacle = state[:, 1:2]
    if radius_px > 0:
        k = 2 * int(radius_px) + 1
        obstacle = F.max_pool2d(obstacle, kernel_size=k, stride=1, padding=int(radius_px))

    pix = normalizer.unnormalize_location(pred_norm.float())
    h, w = obstacle.shape[-2:]
    row = pix[:, 0]
    col = pix[:, 1]
    gx = (2.0 * col / max(w - 1, 1)) - 1.0
    gy = (2.0 * row / max(h - 1, 1)) - 1.0
    grid = torch.stack([gx, gy], dim=-1).view(-1, 1, 1, 2)
    hit = F.grid_sample(
        obstacle.float(),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).view(-1)
    out = (
        F.relu(-row)
        + F.relu(row - (h - 1))
        + F.relu(-col)
        + F.relu(col - (w - 1))
    ) / float(max(h, w))
    return (hit + 0.05 * out).mean()


def main():
    fine_ckpt, out_dir = sys.argv[1], sys.argv[2]
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    batch_size = int(sys.argv[5]) if len(sys.argv) > 5 else 96
    size = int(sys.argv[6]) if len(sys.argv) > 6 else None
    num_workers = int(sys.argv[7]) if len(sys.argv) > 7 else None
    block_penalty = float(sys.argv[8]) if len(sys.argv) > 8 else 0.0
    block_radius_px = int(sys.argv[9]) if len(sys.argv) > 9 else 1
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(Path(fine_ckpt).parent / "config.yaml")
    cfg.data.sample_length = int(cfg.data.get("n_steps", 91)) - 1  # long: far goals
    cfg.data.batch_size = batch_size
    if size is not None:
        cfg.data.size = size
    if num_workers is not None:
        cfg.data.num_workers = num_workers
        cfg.data.pin_mem = False
        cfg.data.persistent_workers = False

    loader, _, data_config, data_pipeline = init_data(
        env_name=cfg.data.env_name,
        cfg_data=OmegaConf.to_container(cfg.data, resolve=True), device=device)
    if data_pipeline is not None:
        print("[subgoal] warming up stream pipeline", flush=True)
        data_pipeline.warm_up()

    jepa, f = build_fine(cfg, data_config, device)
    info = load_checkpoint(Path(fine_ckpt), jepa, optimizer=None, scheduler=None,
                           device=device, strict=False)
    jepa.eval()
    for p in jepa.parameters():
        p.requires_grad_(False)

    subgoal = SubgoalPredictor(f).to(device)
    opt = AdamW(subgoal.parameters(), lr=1e-3, weight_decay=1e-5)
    normalizer = loader.dataset.normalizer
    print(
        f"[subgoal] f={f} N={N} epochs={epochs} block_penalty={block_penalty:g} "
        f"block_radius_px={block_radius_px} | predicts the next A* waypoint",
        flush=True,
    )

    for epoch in range(epochs):
        t0 = time.time(); tot = 0.0; mse_tot = 0.0; block_tot = 0.0; nb = 0
        for x, a, loc, _, _ in loader:
            x = x.to(device, dtype=torch.float32, non_blocking=True)
            loc = loc.to(device, dtype=torch.float32, non_blocking=True)  # [B,2,T] normalized positions
            B, _, T = loc.shape
            with torch.no_grad():
                z = jepa.encode(x)                              # [B,f,T,1,1]
            z_flat = z.permute(0, 2, 1, 3, 4).reshape(B * T, f)  # [B*T,f]
            goal = loc[:, :, -1:].expand(B, 2, T).permute(0, 2, 1).reshape(B * T, 2)
            idx = torch.clamp(torch.arange(T, device=device) + N, max=T - 1)
            label = loc[:, :, idx].permute(0, 2, 1).reshape(B * T, 2)
            pred = subgoal(z_flat, goal)
            mse_loss = F.mse_loss(pred, label)
            block_loss = pred.new_tensor(0.0)
            if block_penalty > 0:
                obs_flat = x.permute(0, 2, 1, 3, 4).reshape(B * T, x.shape[1], x.shape[3], x.shape[4])
                block_loss = observed_obstacle_penalty(
                    pred, obs_flat, normalizer, radius_px=block_radius_px
                )
            loss = mse_loss + block_penalty * block_loss
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
            mse_tot += mse_loss.item()
            block_tot += block_loss.item()
            nb += 1
        print(
            f"[subgoal] epoch {epoch} {time.time()-t0:.0f}s "
            f"loss={tot/max(nb,1):.5f} mse={mse_tot/max(nb,1):.5f} "
            f"block={block_tot/max(nb,1):.5f}",
            flush=True,
        )
        torch.save(
            {
                "subgoal": subgoal.state_dict(),
                "N": N,
                "f": f,
                "block_penalty": block_penalty,
                "block_radius_px": block_radius_px,
            },
            os.path.join(out_dir, "subgoal.pth.tar"),
        )
    if data_pipeline is not None:
        data_pipeline.shutdown()
    print(f"[subgoal] DONE -> {out_dir}/subgoal.pth.tar", flush=True)


if __name__ == "__main__":
    main()
