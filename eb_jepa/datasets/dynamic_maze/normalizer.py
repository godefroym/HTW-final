"""Normalizer for the dynamic-maze 4-channel observation."""

import torch


class DynamicMazeNormalizer:
    """Normalizes dynamic-maze states and pixel locations.

    Observation channels are:
      0. agent dot
      1. observed walls under fog-of-war
      2. unknown/fog mask
      3. observed dynamic-door mask
    """

    def __init__(self, img_size: int = 63):
        self.img_size = img_size
        mid = (img_size - 1) / 2.0
        std = img_size / (12.0 ** 0.5)
        self.location_mean = torch.tensor([mid, mid])
        self.location_std = torch.tensor([std, std])

        self.state_mean = torch.tensor([0.05, 0.35, 0.55, 0.05])
        self.state_std = torch.tensor([0.10, 0.50, 0.50, 0.25])

    def min_max_normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        if len(state.shape) >= 3:
            state = state - state.amin(dim=(-2, -1), keepdim=True)
            state = state / (state.amax(dim=(-2, -1), keepdim=True) + 1e-6)
        else:
            state = state - state.amin(dim=-1, keepdim=True)
            state = state / (state.amax(dim=-1, keepdim=True) + 1e-6)
        return state

    def _stats_for(self, state: torch.Tensor):
        ch = state.shape[-3]
        mean = self.state_mean.to(state.device)
        std = self.state_std.to(state.device)
        if ch != mean.numel():
            if ch < mean.numel():
                mean = mean[:ch]
                std = std[:ch]
            else:
                pad = ch - mean.numel()
                mean = torch.cat([mean, mean[-1:].repeat(pad)])
                std = torch.cat([std, std[-1:].repeat(pad)])
        return mean.view(-1, 1, 1), std.view(-1, 1, 1)

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        state = self.min_max_normalize_state(state)
        mean, std = self._stats_for(state)
        return (state - mean) / (std + 1e-6)

    def unnormalize_state(self, state: torch.Tensor) -> torch.Tensor:
        mean, std = self._stats_for(state)
        return state * std + mean

    def normalize_location(self, location: torch.Tensor) -> torch.Tensor:
        return (location - self.location_mean.to(location.device)) / (
            self.location_std.to(location.device) + 1e-6
        )

    def unnormalize_location(self, location: torch.Tensor) -> torch.Tensor:
        return location * self.location_std.to(location.device) + self.location_mean.to(
            location.device
        )

    def unnormalize_mse(self, mse):
        return mse * (self.location_std.mean().to(mse.device) ** 2)
