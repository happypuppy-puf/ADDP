"""Shared ADDP routing implementation used by every dataset.

The forward pass follows the paper: a sample-level roughness index controls
patch scoring, shared-core retention, and the smooth worst-view temperature.
Hard sorting and integer retention are used in the forward pass and are kept
outside the gradient path. Score and core calibrators are trained with the
continuous auxiliary objectives defined in ``calibration_auxiliary_loss``.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


class ADDPCalibration(nn.Module):
    """Constrained learnable mappings driven by the shared roughness index."""

    def __init__(
        self,
        rho_min: float = 0.2,
        rho_max: float = 0.7,
        tau_min: float = 0.3,
        tau_max: float = 2.5,
    ) -> None:
        super().__init__()
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.tau_min = tau_min
        self.tau_max = tau_max

        self.a_high = nn.Parameter(torch.tensor([math.log(0.5 / 0.3), 0.0]))
        self.a_low = nn.Parameter(torch.tensor([0.0, 0.0, math.log(0.5)]))
        self.a_recency = nn.Parameter(torch.tensor(_inverse_softplus(0.3)))

        slope = 2.0
        self.a_core = nn.Parameter(torch.tensor(0.5 * slope))
        self.b_core = nn.Parameter(torch.tensor(_inverse_softplus(slope)))
        self.a_temperature = nn.Parameter(torch.tensor(0.5 * slope))
        self.b_temperature = nn.Parameter(torch.tensor(_inverse_softplus(slope)))

    def high_weights(self) -> Tensor:
        return F.softmax(self.a_high, dim=-1)

    def low_weights(self) -> Tensor:
        return F.softmax(self.a_low, dim=-1)

    def recency_weight(self) -> Tensor:
        return F.softplus(self.a_recency)

    def core_ratio(self, gamma: Tensor) -> Tensor:
        value = torch.sigmoid(self.a_core - F.softplus(self.b_core) * gamma)
        return self.rho_min + (self.rho_max - self.rho_min) * value

    def temperature(self, gamma: Tensor) -> Tensor:
        value = torch.sigmoid(
            self.a_temperature - F.softplus(self.b_temperature) * gamma
        )
        return self.tau_min + (self.tau_max - self.tau_min) * value

    def describe(self) -> dict:
        device = self.a_core.device
        grid = torch.linspace(0.0, 1.0, 5, device=device)
        with torch.no_grad():
            high = self.high_weights().cpu().tolist()
            low = self.low_weights().cpu().tolist()
            recency = self.recency_weight().item()
            core = self.core_ratio(grid).cpu().tolist()
            temperature = self.temperature(grid).cpu().tolist()
            return {
                "high_weights": high,
                "low_weights": low,
                "recency_weight": recency,
                "core_curve": core,
                "temperature_curve": temperature,
                "alpha_h": high,
                "alpha_l": low,
                "lambda_u": recency,
                "rho_curve": core,
                "tau_curve": temperature,
                "b_c": F.softplus(self.b_core).item(),
                "b_tau": F.softplus(self.b_temperature).item(),
                "a_c": self.a_core.item(),
                "a_tau": self.a_temperature.item(),
            }


def _roughness(values: Tensor, masks: Tensor, scale: float) -> Tensor:
    per_variable = []
    for variable in range(values.shape[-1]):
        sequence = values[:, variable][masks[:, variable] > 0]
        if sequence.numel() > 1:
            variation = (sequence[1:] - sequence[:-1]).abs().sum()
            denominator = sequence.std(unbiased=True) * sequence.numel() + 1e-2
            per_variable.append(variation / denominator)
    if not per_variable:
        return values.new_zeros(())
    return (torch.stack(per_variable).mean() / scale).clamp(0.0, 1.0)


def apply_addp_routing_calibrated(
    values: Tensor,
    masks: Tensor,
    global_mean: Tensor,
    calibration: ADDPCalibration,
    conservative_ratio: float = 0.8,
    aggressive_ratio: float = 0.4,
    roughness_scale: float = 1.5,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Build the conservative and aggressive training views.

    The complete shared core is retained in both views. If its learned size is
    larger than a nominal branch budget, the core is not truncated.
    """

    batch_size, patch_count, patch_length, variable_count = values.shape
    flat_values = values.reshape(batch_size, patch_count * patch_length, variable_count)
    flat_masks = masks.reshape(batch_size, patch_count * patch_length, variable_count)
    observed_sum = (flat_values * flat_masks).sum(dim=1)
    observed_count = flat_masks.sum(dim=1)
    fallback = global_mean.to(values.device).reshape(1, variable_count)
    local_mean = torch.where(
        observed_count > 0,
        observed_sum / observed_count.clamp_min(1.0),
        fallback,
    )

    high_weights = calibration.high_weights()
    low_weights = calibration.low_weights()
    recency_weight = calibration.recency_weight()
    conservative_target = max(1, int(patch_count * conservative_ratio))
    aggressive_target = max(1, int(patch_count * aggressive_ratio))

    conservative_masks = []
    aggressive_masks = []
    temperatures = []
    core_ratios = []
    all_scores = []
    gammas = []

    for sample in range(batch_size):
        gamma = _roughness(flat_values[sample], flat_masks[sample], roughness_scale)
        gamma = gamma.detach()
        gammas.append(gamma)
        core_ratio = calibration.core_ratio(gamma)
        temperature = calibration.temperature(gamma)
        core_ratios.append(core_ratio)
        temperatures.append(temperature)

        score_parts = []
        for patch in range(patch_count):
            patch_mask = masks[sample, patch] > 0
            observed = values[sample, patch][patch_mask]
            if observed.numel() == 0:
                score_parts.append(values.new_zeros(()))
                continue
            density = patch_mask.float().mean()
            energy = observed.var(unbiased=True) if observed.numel() > 1 else observed.new_zeros(())
            reference = local_mean[sample].mean()
            anomaly = (observed.mean() - reference).abs()
            continuity = 1.0 / (energy + 1e-3)
            decay = 0.1 * (1.0 - gamma)
            recency = torch.exp(
                -decay * values.new_tensor(float(patch_count - 1 - patch))
            )
            high_score = high_weights[0] * energy + high_weights[1] * density
            low_score = (
                low_weights[0] * anomaly
                + low_weights[1] * continuity
                + low_weights[2] * density
            )
            score_parts.append(
                gamma * high_score
                + (1.0 - gamma) * low_score
                + recency_weight * recency
            )
        scores = torch.stack(score_parts)
        all_scores.append(scores)
        order = torch.argsort(scores, descending=True)
        core_count = max(1, int(torch.floor(core_ratio.detach() * patch_count).item()))
        core = order[:core_count]
        remainder = order[core_count:]
        if remainder.numel() > 0:
            remainder = remainder[torch.randperm(remainder.numel(), device=values.device)]

        conservative = torch.cat(
            [core, remainder[: max(0, conservative_target - core_count)]]
        )
        aggressive = torch.cat(
            [core, remainder[: max(0, aggressive_target - core_count)]]
        )
        hard_conservative = values.new_zeros(patch_count).scatter(0, conservative, 1.0)
        hard_aggressive = values.new_zeros(patch_count).scatter(0, aggressive, 1.0)
        # Memberships are discrete forward decisions.  In accordance with the
        # paper, no straight-through estimator is used for ranking or top-k.
        conservative_masks.append(hard_conservative)
        aggressive_masks.append(hard_aggressive)
    conservative_mask = torch.stack(conservative_masks)
    aggressive_mask = torch.stack(aggressive_masks)
    gamma_tensor = torch.stack(gammas)
    replacement = (
        (1.0 - gamma_tensor).reshape(batch_size, 1, 1, 1)
        * local_mean.reshape(batch_size, 1, 1, variable_count)
    )
    conservative_view = (
        conservative_mask.reshape(batch_size, patch_count, 1, 1) * values
        + (1.0 - conservative_mask.reshape(batch_size, patch_count, 1, 1)) * replacement
    )
    aggressive_view = (
        aggressive_mask.reshape(batch_size, patch_count, 1, 1) * values
        + (1.0 - aggressive_mask.reshape(batch_size, patch_count, 1, 1)) * replacement
    )

    hard_conservative = conservative_mask
    hard_aggressive = aggressive_mask
    topology_replacement = (1.0 - gamma_tensor).reshape(batch_size, 1)
    conservative_topology = (
        hard_conservative + (1.0 - hard_conservative) * topology_replacement
    )
    aggressive_topology = (
        hard_aggressive + (1.0 - hard_aggressive) * topology_replacement
    )
    return (
        conservative_view,
        aggressive_view,
        conservative_topology,
        aggressive_topology,
        torch.stack(temperatures),
        torch.stack(core_ratios),
        torch.stack(all_scores),
        hard_conservative,
        hard_aggressive,
    )


def calibration_auxiliary_loss(
    patch_scores: Tensor,
    conservative_membership: Tensor,
    core_ratios: Tensor,
    *,
    score_weight: float,
    core_weight: float = 1e-4,
    margin: float = 0.1,
    core_target: float = 0.45,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Return the paper's differentiable score and core calibration losses.

    The hard kept/dropped memberships are treated as fixed labels.  Gradients
    therefore reach the scoring calibrator only through ``patch_scores`` and
    the shared-core calibrator only through the continuous ``core_ratios``.
    """

    keep = conservative_membership.detach().to(patch_scores.dtype)
    drop = 1.0 - keep
    kept_count = keep.sum(dim=1).clamp_min(1.0)
    dropped_count = drop.sum(dim=1)

    kept_mean = (patch_scores * keep).sum(dim=1) / kept_count
    dropped_mean = (patch_scores * drop).sum(dim=1) / dropped_count.clamp_min(1.0)
    applicable = dropped_count > 0
    if applicable.any():
        score_loss = F.relu(
            margin + dropped_mean[applicable] - kept_mean[applicable]
        ).mean()
    else:
        score_loss = patch_scores.sum() * 0.0

    core_loss = (core_ratios - core_target).square().mean()
    weighted_score = score_weight * score_loss
    weighted_core = core_weight * core_loss
    return weighted_score + weighted_core, weighted_score, weighted_core
