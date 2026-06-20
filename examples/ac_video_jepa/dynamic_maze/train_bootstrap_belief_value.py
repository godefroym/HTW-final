"""Train a recurrent belief value head on a frozen dynamic-maze JEPA.

This is the memory bootstrap variant: data collection stays A*-free
(`local_goal_frontier`), the JEPA world model is frozen, and a small LSTM learns
an episode belief state from pooled JEPA latents plus previous actions. The
goal-conditioned value head is then trained by hindsight TD on both real latents
and the frozen world model's rollouts.

Run:
  python -m examples.ac_video_jepa.dynamic_maze.train_bootstrap_belief_value \
    --wm-ckpt <bootstrap_wm/latest.pth.tar> --out-dir <run_dir>
"""

import argparse
import json
import os
from pathlib import Path
from time import time

import torch
from omegaconf import OmegaConf
from torch.optim import AdamW
from tqdm import tqdm

from eb_jepa.datasets.utils import init_data
from eb_jepa.schedulers import CosineWithWarmup
from eb_jepa.state_decoder import BeliefLSTM, BeliefValueHead
from eb_jepa.training_utils import load_checkpoint
from examples.ac_video_jepa.maze.maze_fine_wm import build_fine


def ema_update(target, online, tau):
    with torch.no_grad():
        for pt, p in zip(target.parameters(), online.parameters()):
            pt.mul_(tau).add_(p.detach(), alpha=1.0 - tau)


def state_dict_clone(module):
    return {k: v.detach().cpu() for k, v in module.state_dict().items()}


def save_belief_checkpoint(
    path,
    jepa,
    source_ckpt,
    belief_lstm,
    value_head,
    epoch,
    step,
    cfg,
    args,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": int(epoch),
        "step": int(step),
        "model_state_dict": state_dict_clone(jepa),
        "belief_lstm_state_dict": state_dict_clone(belief_lstm),
        "belief_value_head_state_dict": state_dict_clone(value_head),
        "belief_hidden_dim": int(args.hidden_dim),
        "belief_num_layers": int(args.num_layers),
        "belief_action_dim": 2,
        "belief_source_wm_ckpt": str(args.wm_ckpt),
        "belief_train_args": vars(args),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    if "xy_head_state_dict" in source_ckpt:
        checkpoint["xy_head_state_dict"] = source_ckpt["xy_head_state_dict"]
    torch.save(checkpoint, path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wm-ckpt", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--size", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--num-gen-workers", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.96)
    parser.add_argument("--ema-tau", type=float, default=0.99)
    parser.add_argument("--horizons", default="1,2,4,8,16,32")
    parser.add_argument("--pairs-per-horizon", type=int, default=12)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wm_ckpt = Path(args.wm_ckpt)
    cfg = OmegaConf.load(wm_ckpt.parent / "config.yaml")

    cfg.data.size = int(args.size)
    cfg.data.batch_size = int(args.batch_size)
    cfg.data.teacher_policy = "local_goal_frontier"
    cfg.data.layout_use_astar_filter = False
    cfg.data.num_workers = 0
    cfg.data.pin_mem = False
    cfg.data.persistent_workers = False
    if cfg.data.get("pipeline") is not None:
        cfg.data.pipeline.mode = "stream"
        cfg.data.pipeline.backend = "cpu"
        cfg.data.pipeline.chunk_size = int(args.chunk_size)
        cfg.data.pipeline.num_gen_workers = int(args.num_gen_workers)

    OmegaConf.save(cfg, out_dir / "config.yaml")
    with (out_dir / "belief_train_args.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader, _, data_config, data_pipeline = init_data(
        cfg.data.env_name,
        cfg_data=OmegaConf.to_container(cfg.data, resolve=True),
        device=device,
    )
    if data_pipeline is not None:
        print("[belief-value] warming data stream", flush=True)
        data_pipeline.warm_up()

    jepa, latent_dim = build_fine(cfg, data_config, device)
    source_ckpt = load_checkpoint(wm_ckpt, jepa, optimizer=None, scheduler=None,
                                  device=device, strict=False)
    jepa.eval()
    for p in jepa.parameters():
        p.requires_grad_(False)

    belief_lstm = BeliefLSTM(latent_dim, action_dim=2, hidden_dim=args.hidden_dim,
                            num_layers=args.num_layers).to(device)
    belief_target = BeliefLSTM(latent_dim, action_dim=2, hidden_dim=args.hidden_dim,
                              num_layers=args.num_layers).to(device)
    belief_target.load_state_dict(belief_lstm.state_dict())
    for p in belief_target.parameters():
        p.requires_grad_(False)

    value_head = BeliefValueHead(args.hidden_dim, latent_dim).to(device)
    value_target = BeliefValueHead(args.hidden_dim, latent_dim).to(device)
    value_target.load_state_dict(value_head.state_dict())
    for p in value_target.parameters():
        p.requires_grad_(False)

    optim = AdamW(
        list(belief_lstm.parameters()) + list(value_head.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineWithWarmup(
        optim,
        total_steps=max(1, args.epochs * len(loader)),
        warmup_ratio=0.1,
    )
    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]
    metrics_path = out_dir / "metrics.tsv"
    with metrics_path.open("w") as f:
        f.write("epoch\tstep\tloss\tseconds\n")

    global_step = 0
    for epoch in range(args.epochs):
        start = time()
        last_loss = 0.0
        pbar = tqdm(enumerate(loader), total=len(loader),
                    desc=f"belief epoch {epoch}/{args.epochs - 1}")
        for idx, (x, a, _loc, _unused1, _unused2) in pbar:
            x = x.to(device, non_blocking=True).float()
            a = a.to(device, non_blocking=True).float()

            with torch.no_grad():
                z_gt = jepa.encode(x).float()
                tw = z_gt.shape[2]
                z_roll, _ = jepa.unroll(
                    x[:, :, :1],
                    a,
                    nsteps=tw - 1,
                    unroll_mode="autoregressive",
                    ctxt_window_time=1,
                    compute_loss=False,
                    return_all_steps=False,
                )
                z_roll = z_roll.float()
                h_gt_targ, _ = belief_target(z_gt, a)
                h_roll_targ, _ = belief_target(z_roll, a)

            h_gt, _ = belief_lstm(z_gt.detach(), a)
            h_roll, _ = belief_lstm(z_roll.detach(), a)
            loss_terms = []

            for horizon in horizons:
                if horizon < 1 or horizon >= tw:
                    continue
                n_pairs = tw - horizon
                if args.pairs_per_horizon > 0 and args.pairs_per_horizon < n_pairs:
                    t_idx = torch.randperm(n_pairs, device=device)[: args.pairs_per_horizon]
                    t_idx = torch.sort(t_idx).values
                else:
                    t_idx = torch.arange(n_pairs, device=device)

                next_idx = t_idx + 1
                goal_idx = t_idx + horizon
                done = torch.full(
                    (h_gt.shape[0], t_idx.shape[0]),
                    1.0 if horizon == 1 else 0.0,
                    device=device,
                    dtype=h_gt.dtype,
                )
                g = z_gt.index_select(2, goal_idx).detach()

                with torch.no_grad():
                    target_real = done + args.gamma * (1.0 - done) * value_target(
                        h_gt_targ.index_select(1, next_idx), g
                    )
                    target_roll = done + args.gamma * (1.0 - done) * value_target(
                        h_roll_targ.index_select(1, next_idx), g
                    )

                loss_terms.append(
                    torch.nn.functional.mse_loss(
                        value_head(h_gt.index_select(1, t_idx), g), target_real
                    )
                )
                loss_terms.append(
                    torch.nn.functional.mse_loss(
                        value_head(h_roll.index_select(1, t_idx), g), target_roll
                    )
                )

            if not loss_terms:
                raise ValueError(f"no valid value horizons for trajectory length {tw}")

            loss = torch.stack(loss_terms).mean()
            optim.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(belief_lstm.parameters()) + list(value_head.parameters()),
                    args.grad_clip,
                )
            optim.step()
            scheduler.step()
            ema_update(belief_target, belief_lstm, args.ema_tau)
            ema_update(value_target, value_head, args.ema_tau)

            last_loss = float(loss.detach().cpu())
            global_step = epoch * len(loader) + idx
            if idx % args.log_every == 0:
                pbar.set_postfix({"loss": f"{last_loss:.5f}"})

        elapsed = time() - start
        with metrics_path.open("a") as f:
            f.write(f"{epoch}\t{global_step}\t{last_loss:.8f}\t{elapsed:.2f}\n")
        save_belief_checkpoint(
            out_dir / "latest.pth.tar",
            jepa,
            source_ckpt,
            belief_lstm,
            value_head,
            epoch,
            global_step,
            cfg,
            args,
        )
        save_belief_checkpoint(
            out_dir / f"e-{epoch}.pth.tar",
            jepa,
            source_ckpt,
            belief_lstm,
            value_head,
            epoch,
            global_step,
            cfg,
            args,
        )
        print(
            f"[belief-value] epoch={epoch}/{args.epochs} loss={last_loss:.5f} "
            f"time={elapsed:.1f}s",
            flush=True,
        )

    if data_pipeline is not None:
        data_pipeline.shutdown()


if __name__ == "__main__":
    main()
