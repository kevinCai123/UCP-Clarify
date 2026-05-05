# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

import dataclasses
import enum
from typing import Callable, Dict, Optional
from transformers.trainer_utils import EvalLoopOutput
from trl import DPOConfig, SFTConfig

@dataclasses.dataclass
class TrainingConfig(DPOConfig):
  """Training configuration."""
  project_id: Optional[str] = None
  location: Optional[str] = None
  staging_bucket: Optional[str] = None
  machine_type: str = ''
  accelerator_type: str = ''
  bf16: bool = False
  batch_size: int = 2
  seed: int = 42
  task: Optional[str] = None
  log_level: str = 'info'
  log_on_each_node: bool = False
  logging_strategy: str = 'epoch'
  do_eval: bool = True
  eval_strategy: str = 'epoch'
  resume_from_checkpoint: Optional[str] = None
  save_strategy: str = 'epoch'
  save_total_limit: int = 1
  load_best_model_at_end: bool = True
  output_dir: str = 'outputs/model_output'
  per_device_train_batch_size: int = 1
  per_device_eval_batch_size: int = 1
  gradient_checkpointing: bool = True
  gradient_checkpointing_kwargs: dict[str, bool] = dataclasses.field(
      default_factory=lambda: {'use_reentrant': False}
  )
  world_size: int = 1
  learning_rate: float = 5.0e-7
  lr_scheduler_type: str = 'cosine'
  remove_unused_columns: bool = False
  target_label: str = 'CLARIFY'
  icl_examples: int = 10
  class_balance: Optional[float] = None
  mixed_precision: bool = False
  is_preference: bool = False
  max_length: Optional[int] = 1024


@dataclasses.dataclass
class ACTConfig(TrainingConfig):
  """Arguments related to the DPO training process itself.
  """

  beta: Optional[float] = 0.05
  experiment_mode: str = 'act_original'
  beta_dpo_alpha: float = 0.6
  beta_dpo_filter_ratio: float = 0.2
  beta_dpo_ema_gamma: float = 0.9
  beta_dpo_min_beta: float = 1.0e-3
  beta_dpo_max_beta: float = 0.15
  beta_dpo_warmup_steps: int = 0
  clarify_loss_weight: float = 1.0
  # Vanilla preference-optimization baselines (act_ipo/act_simpo/act_orpo).
  # Ignored by act_original and the beta-DPO modes.
  simpo_beta: float = 2.0
  simpo_gamma: float = 1.0
  orpo_lambda: float = 0.1
  num_train_epochs: int = 2
  logging_first_step: bool = True
  max_prompt_length: Optional[int] = 1024
  optim: Optional[str] = 'rmsprop'
  loss_type: Optional[str] = 'sigmoid'
  metric_function: Optional[str] = None
  sample_frequency: int = 1

  def __post_init__(self):
    super().__post_init__()
    lt = getattr(self, 'loss_type', None) or 'sigmoid'
    if isinstance(lt, str):
      self.loss_types = [lt]
    elif isinstance(lt, list):
      self.loss_types = lt


@dataclasses.dataclass
class ACTInitializationConfig(TrainingConfig, SFTConfig):
  """Arguments related to the SFT training process itself."""
  num_train_epochs: int = 8
  logging_first_step: bool = True
  max_seq_length: Optional[int] = 2048
  optim: Optional[str] = 'adamw_torch'
  loss_type: Optional[str] = 'nll'
  packing: bool = True

  def __post_init__(self):
    super().__post_init__()
    if isinstance(self.loss_type, list):
      self.loss_type = self.loss_type[0] if self.loss_type else 'nll'
