# coding=utf-8
# Copyright 2026 Songlin Cai.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


ACT_EXPERIMENT_MODES = {
    "act_original",
    "act_dynamic_beta",
    "act_beta_filter",
    "act_beta_dpo_full",
    "act_instance_beta_ablation",
    # Vanilla preference-optimization baselines handled inline in
    # ACTTrainer._compute_loss. They reuse UCP-Clarify rollout + clarify_loss_weight
    # but swap the pairwise loss for IPO / SimPO / ORPO.
    "act_ipo",
    "act_simpo",
    "act_orpo",
}


@dataclass(frozen=True)
class BetaDPOModeConfig:
  experiment_mode: str
  base_beta: float
  alpha: float
  filter_ratio: float
  ema_gamma: float
  min_beta: float
  max_beta: float = 0.15
  warmup_steps: int = 0

  @property
  def use_dynamic_beta(self) -> bool:
    return self.experiment_mode in {
        "act_dynamic_beta",
        "act_beta_dpo_full",
        "act_instance_beta_ablation",
    }

  @property
  def use_filtering(self) -> bool:
    return self.experiment_mode in {
        "act_beta_filter",
        "act_beta_dpo_full",
    }

  @property
  def use_instance_beta(self) -> bool:
    return self.experiment_mode == "act_instance_beta_ablation"

  def validate(self) -> None:
    if self.experiment_mode not in ACT_EXPERIMENT_MODES:
      raise ValueError(
          f"Unsupported UCP-Clarify experiment mode '{self.experiment_mode}'. "
          f"Expected one of {sorted(ACT_EXPERIMENT_MODES)}."
      )
    if not 0.0 <= self.filter_ratio < 1.0:
      raise ValueError(
          f"beta_dpo_filter_ratio must be in [0, 1). Got {self.filter_ratio}."
      )
    if not 0.0 <= self.ema_gamma < 1.0:
      raise ValueError(
          f"beta_dpo_ema_gamma must be in [0, 1). Got {self.ema_gamma}."
      )
    if self.base_beta <= 0.0:
      raise ValueError(f"beta must be > 0. Got {self.base_beta}.")
    if self.min_beta <= 0.0:
      raise ValueError(
          f"beta_dpo_min_beta must be > 0. Got {self.min_beta}."
      )
    if self.max_beta < self.min_beta:
      raise ValueError(
          f"beta_dpo_max_beta ({self.max_beta}) must be >= "
          f"beta_dpo_min_beta ({self.min_beta})."
      )


@dataclass
class BetaDPOComputation:
  losses: torch.FloatTensor
  chosen_rewards: torch.FloatTensor
  rejected_rewards: torch.FloatTensor
  discrepancy: torch.FloatTensor
  local_filter_weights: torch.FloatTensor
  global_filter_weights: torch.FloatTensor
  beta_used: torch.FloatTensor
  batch_mean: torch.FloatTensor
  batch_std: torch.FloatTensor
  running_mean: torch.FloatTensor
  running_std: torch.FloatTensor
  selected_mean: torch.FloatTensor
  actual_filter_ratio: torch.FloatTensor


def compute_discrepancy(
    chosen_logps: torch.FloatTensor,
    rejected_logps: torch.FloatTensor,
    ref_chosen_logps: torch.FloatTensor,
    ref_rejected_logps: torch.FloatTensor,
    reference_free: bool,
) -> torch.FloatTensor:
  if reference_free:
    return chosen_logps - rejected_logps
  return (
      chosen_logps
      - ref_chosen_logps
      - rejected_logps
      + ref_rejected_logps
  )


def update_running_moments(
    values: torch.FloatTensor,
    running_mean: torch.FloatTensor,
    running_std: torch.FloatTensor,
    gamma: float,
    initialized: bool,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, bool]:
  batch_mean = values.mean()
  batch_std = values.std(unbiased=False)

  if not initialized:
    return batch_mean.detach(), batch_std.detach(), True

  new_mean = running_mean * gamma + batch_mean.detach() * (1.0 - gamma)
  new_std = running_std * gamma + batch_std.detach() * (1.0 - gamma)
  return new_mean, new_std, True


def build_soft_filter_weights(
    values: torch.FloatTensor,
    running_mean: torch.FloatTensor,
    running_std: torch.FloatTensor,
    filter_ratio: float,
    protected_mask: Optional[torch.BoolTensor] = None,
) -> torch.FloatTensor:
  """Soft filter: downweight outliers instead of hard dropping them.

  Returns per-sample weights in [1 - filter_ratio, 1.0].  Samples close to
  the running mean keep weight ~1.0; extreme outliers are reduced toward
  (1 - filter_ratio).  Protected (CLARIFY) samples always get weight 1.0.
  """
  num_values = values.numel()
  if num_values == 0:
    return torch.ones_like(values)

  if filter_ratio <= 0.0:
    return torch.ones(num_values, dtype=torch.float, device=values.device)

  safe_std = running_std.clamp(min=1e-6)
  raw = torch.exp(-0.5 * ((values - running_mean) / safe_std).pow(2))

  if not torch.isfinite(raw).all() or raw.max() <= 0:
    return torch.ones(num_values, dtype=torch.float, device=values.device)

  normalized = raw / raw.max()
  weights = 1.0 - filter_ratio * (1.0 - normalized)

  if protected_mask is not None and protected_mask.any():
    weights = weights.clone()
    weights[protected_mask] = 1.0

  return weights


def compute_beta_dpo(
    config: BetaDPOModeConfig,
    local_discrepancy: torch.FloatTensor,
    global_discrepancy: torch.FloatTensor,
    running_mean: torch.FloatTensor,
    running_std: torch.FloatTensor,
    process_index: int,
    num_processes: int,
    global_step: int,
    chosen_logps: torch.FloatTensor,
    rejected_logps: torch.FloatTensor,
    ref_chosen_logps: torch.FloatTensor,
    ref_rejected_logps: torch.FloatTensor,
    reference_free: bool,
    protected_mask: Optional[torch.BoolTensor] = None,
) -> BetaDPOComputation:
  batch_mean = global_discrepancy.mean()
  batch_std = global_discrepancy.std(unbiased=False)

  if config.use_filtering:
    global_filter_weights = build_soft_filter_weights(
        global_discrepancy,
        running_mean,
        running_std,
        config.filter_ratio,
        protected_mask=protected_mask,
    )
  else:
    global_filter_weights = torch.ones_like(global_discrepancy)

  local_batch_size = local_discrepancy.shape[0]
  start = process_index * local_batch_size
  end = start + local_batch_size
  local_filter_weights = global_filter_weights[start:end]
  weight_sum = global_filter_weights.sum().clamp(min=1e-6)
  selected_mean = (global_discrepancy * global_filter_weights).sum() / weight_sum

  in_warmup = global_step < config.warmup_steps
  if config.use_dynamic_beta and not in_warmup:
    safe_std = running_std.clamp(min=1e-6)
    if config.use_instance_beta:
      normalized_gap = (local_discrepancy - running_mean) / safe_std
      beta_used = config.base_beta * (
          1.0 + config.alpha * normalized_gap
      )
      beta_used = beta_used.clamp(
          min=config.min_beta, max=config.max_beta,
      ).detach()
    else:
      normalized_gap = (selected_mean - running_mean) / safe_std
      beta_scalar = config.base_beta * (
          1.0 + config.alpha * normalized_gap
      )
      beta_scalar = beta_scalar.clamp(
          min=config.min_beta, max=config.max_beta,
      ).detach()
      beta_used = torch.full_like(local_discrepancy, beta_scalar)
  else:
    beta_used = torch.full_like(local_discrepancy, config.base_beta)

  losses = -F.logsigmoid(beta_used * local_discrepancy)
  reward_ref_chosen = (
      torch.zeros_like(ref_chosen_logps) if reference_free else ref_chosen_logps
  )
  reward_ref_rejected = (
      torch.zeros_like(ref_rejected_logps)
      if reference_free
      else ref_rejected_logps
  )
  chosen_rewards = (
      config.base_beta * (chosen_logps - reward_ref_chosen)
  ).detach()
  rejected_rewards = (
      config.base_beta * (rejected_logps - reward_ref_rejected)
  ).detach()

  actual_filter_ratio = 1.0 - global_filter_weights.mean()

  return BetaDPOComputation(
      losses=losses,
      chosen_rewards=chosen_rewards,
      rejected_rewards=rejected_rewards,
      discrepancy=local_discrepancy,
      local_filter_weights=local_filter_weights,
      global_filter_weights=global_filter_weights,
      beta_used=beta_used,
      batch_mean=batch_mean.detach(),
      batch_std=batch_std.detach(),
      running_mean=running_mean.detach(),
      running_std=running_std.detach(),
      selected_mean=selected_mean.detach(),
      actual_filter_ratio=actual_filter_ratio.detach(),
  )
