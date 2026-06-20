import torch
from torch import nn


def _pooled_latent(x):
    """Return latent tokens as [B, T, C] from [B, C, T, H, W]."""
    if x.ndim != 5:
        raise ValueError(f"expected latent [B,C,T,H,W], got {tuple(x.shape)}")
    return x.mean(dim=(3, 4)).permute(0, 2, 1)


class BeliefLSTM(nn.Module):
    """Recurrent belief state over pooled JEPA latents and previous actions.

    This is deliberately small: the world model still provides the visual/action
    latent dynamics, while the LSTM carries episode memory for partial
    observability (fog-of-war, previously visited corridors, recently seen
    doors). Input at time t is pooled z_t plus the action that led to z_t.
    """

    def __init__(self, latent_dim, action_dim=2, hidden_dim=256, num_layers=1):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.lstm = nn.LSTM(
            input_size=self.latent_dim + self.action_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
        )

    def _prev_actions(self, actions, timesteps, device, dtype):
        if actions is None:
            return torch.zeros(
                (1, timesteps, self.action_dim), device=device, dtype=dtype
            )
        if actions.ndim != 3:
            raise ValueError(f"expected actions [B,A,T], got {tuple(actions.shape)}")
        b = actions.shape[0]
        prev = torch.zeros((b, timesteps, self.action_dim), device=device, dtype=dtype)
        usable = min(max(timesteps - 1, 0), actions.shape[2])
        if usable > 0:
            prev[:, 1 : usable + 1] = actions[:, : self.action_dim, :usable].permute(
                0, 2, 1
            ).to(dtype=dtype)
        return prev

    def forward(self, latents, actions=None, hidden=None):
        z = _pooled_latent(latents)
        prev_a = self._prev_actions(actions, z.shape[1], z.device, z.dtype)
        if prev_a.shape[0] == 1 and z.shape[0] != 1:
            prev_a = prev_a.expand(z.shape[0], -1, -1)
        out, hidden = self.lstm(torch.cat([z, prev_a], dim=-1), hidden)
        return out, hidden

    def step(self, latent, prev_action=None, hidden=None):
        """Single recurrent update.

        Args:
            latent: [B,C,1,H,W] or [B,C,H,W]
            prev_action: [B,A] action that led to this latent
        Returns:
            belief: [B,H], hidden tuple for the next step.
        """
        if latent.ndim == 4:
            latent = latent.unsqueeze(2)
        z = _pooled_latent(latent)
        b = z.shape[0]
        if prev_action is None:
            prev_action = torch.zeros(
                (b, self.action_dim), device=z.device, dtype=z.dtype
            )
        else:
            prev_action = prev_action[:, : self.action_dim].to(device=z.device, dtype=z.dtype)
        x = torch.cat([z[:, :1], prev_action.unsqueeze(1)], dim=-1)
        out, hidden = self.lstm(x, hidden)
        return out[:, -1], hidden


class BeliefValueHead(nn.Module):
    """Goal-conditioned scalar value V(h_state, z_goal) for BeliefLSTM states."""

    def __init__(self, belief_dim, goal_dim, hidden=512):
        super().__init__()
        self.belief_dim = int(belief_dim)
        self.goal_dim = int(goal_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.belief_dim + self.goal_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, belief, goal):
        """
        Args:
            belief: [B, T, H] or [B, H]
            goal:   [B or 1, C, 1, h, w]
        Returns:
            value: [B, T] in (0, 1)
        """
        if belief.ndim == 2:
            belief = belief.unsqueeze(1)
        if belief.ndim != 3:
            raise ValueError(f"expected belief [B,T,H], got {tuple(belief.shape)}")
        bs, t, _ = belief.shape
        g = _pooled_latent(goal)
        if g.shape[1] == 1:
            g = g.expand(bs, t, self.goal_dim)
        elif g.shape[1] != t:
            raise ValueError(
                f"goal has {g.shape[1]} time steps but belief has {t}"
            )
        v = self.mlp(torch.cat([belief, g], dim=-1)).squeeze(-1)
        return torch.sigmoid(v)


class GoalValueHead(nn.Module):
    """Goal-conditioned scalar value V(z_state, z_goal) — TD-MPC style.

    Replaces the crude "distance-in-latent-space" planning cost with a *learned*
    value (Hansen et al., TD-MPC 2022/2024): the head maps a state latent and a
    goal latent to a scalar in (0, 1) interpreted as the discounted return-to-goal
    (≈ ``gamma ** steps_to_goal``). Trained by TD on the world model's own
    rollouts (see ``examples/ac_video_jepa/main.py``); at planning time the MPC
    objective MAXIMISES this value, so the planner optimises a quantity that
    correlates with task success rather than raw representation distance.

    Mirrors ``MLPXYHead``'s pooled-latent interface: latents are [B, C, T, h, w]
    with h=w=1 for the impala encoder, so spatial mean-pooling is a no-op there.
    """

    def __init__(self, input_shape, hidden=512):  # input_shape = C (channel dim)
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * input_shape, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, state, goal):
        """
        Args:
            state: [B, C, T, h, w]
            goal:  [B or 1, C, 1, h, w]
        Returns:
            value: [B, T] in (0, 1)
        """
        bs, c, t, h, w = state.shape
        s = state.mean(dim=(3, 4)).permute(0, 2, 1)        # [B, T, C]
        g = goal.mean(dim=(3, 4)).permute(0, 2, 1)         # [B or 1, 1, C]
        g = g.expand(bs, t, c)                             # [B, T, C]
        feat = torch.cat([s, g], dim=-1)                   # [B, T, 2C]
        v = self.mlp(feat).squeeze(-1)                     # [B, T]
        return torch.sigmoid(v)


class MLPXYHead(nn.Module):
    """A head to recover the xy location from features."""

    def __init__(self, input_shape, normalizer=None):  # input_shape = (C, H, W)
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_shape, 512), nn.ReLU(inplace=True), nn.Linear(512, 2)
        )
        self.normalizer = normalizer

    def forward(self, x):
        """
        Args:
            x: [B, C, T, H, W]
        Returns:
            pred: [B, 2, T]
        """
        bs, c, t, h, w = x.shape

        x = x.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
        x = x.reshape(bs * t, c, h, w)  # [B*T, C, H, W]

        x = x.squeeze(-1).squeeze(-1)  # [B*T, C]

        pred = self.mlp(x)

        pred = pred.view(bs, t, 2).permute(0, 2, 1)

        return pred
