# coding=utf-8
# Copyright 2025 The Google Research Authors.
# Modifications Copyright 2026 Songlin Cai.
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

"""UCP-Clarify trainer built on top of the current TRL DPOTrainer API."""

from collections import defaultdict
import logging
import os
from typing import Literal

from ucp_clarify.metrics.base_metrics import BaseMetrics
from ucp_clarify.simulation.simulator import Simulator
from ucp_clarify.trainer.beta_dpo import (
    BetaDPOModeConfig,
    compute_beta_dpo,
    compute_discrepancy,
    update_running_moments,
)
from datasets import Dataset
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase
from trl import DPOTrainer

# --- Compatibility shims for trl >= 0.15 -----------------------------------
# selective_log_softmax moved from trl.trainer.dpo_trainer to trl.trainer.utils.
# entropy_from_logits / disable_gradient_checkpointing were removed from public
# trl API. Provide safe fallbacks so training works across trl versions.
try:
  from trl.trainer.dpo_trainer import selective_log_softmax  # trl < 0.15
except ImportError:
  try:
    from trl.trainer.utils import selective_log_softmax  # trl >= 0.15
  except ImportError:
    def selective_log_softmax(logits, index):
      log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
      return log_probs.gather(-1, index.unsqueeze(-1)).squeeze(-1)

try:
  from trl.trainer.dpo_trainer import entropy_from_logits  # trl < 0.13
except ImportError:
  try:
    from trl.core import entropy_from_logits  # older trl
  except ImportError:
    def entropy_from_logits(logits):
      log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
      probs = log_probs.exp()
      return -(probs * log_probs).sum(dim=-1)

try:
  from trl.trainer.dpo_trainer import disable_gradient_checkpointing  # trl < 0.13
except ImportError:
  from contextlib import contextmanager

  @contextmanager
  def disable_gradient_checkpointing(model):
    was_enabled = False
    if hasattr(model, "is_gradient_checkpointing") and model.is_gradient_checkpointing:
      was_enabled = True
      model.gradient_checkpointing_disable()
    try:
      yield
    finally:
      if was_enabled:
        model.gradient_checkpointing_enable()
# ---------------------------------------------------------------------------

from peft import PeftModel

def _is_peft_model(model):
  return isinstance(model, PeftModel)

def _use_adapter(model, adapter_name=None):
  if adapter_name is None:
    return model.disable_adapter()
  return model.set_adapter(adapter_name)


class _ACTMetadataCollator:

  def __init__(self, base_collator, metadata_keys):
    self.base_collator = base_collator
    self.metadata_keys = metadata_keys

  def __call__(self, features):
    collated = self.base_collator(features)
    for key in self.metadata_keys:
      if features and key in features[0]:
        collated[key] = [feature[key] for feature in features]
    return collated


class ACTTrainer(DPOTrainer):

  def _load_optimizer_and_scheduler(self, checkpoint):
    """Resume training even if the saved optimizer state is incompatible."""
    try:
      return super()._load_optimizer_and_scheduler(checkpoint)
    except ValueError as exc:
      if checkpoint is None or 'parameter group' not in str(exc):
        raise

      logging.warning(
          (
              'Skipping optimizer state restore from %s because the saved '
              'parameter groups are incompatible with the current optimizer. '
              'Continuing with a fresh optimizer. Error: %s'
          ),
          checkpoint,
          exc,
      )

      scheduler_path = os.path.join(checkpoint, 'scheduler.pt')
      if self.lr_scheduler is not None and os.path.isfile(scheduler_path):
        self.lr_scheduler.load_state_dict(
            torch.load(scheduler_path, weights_only=True)
        )
        logging.info('Loaded scheduler state from %s.', scheduler_path)
      else:
        logging.warning(
            'No scheduler state found at %s. Continuing with a fresh scheduler.',
            scheduler_path,
        )

  def __init__(
      self,
      model,
      ref_model,
      args,
      action_model,
      user_simulator,
      intent_summarization_model,
      train_dataset,
      eval_dataset,
      tokenizer: PreTrainedTokenizerBase,
      metrics: BaseMetrics,
      beta=0.1,
      label_smoothing=0,
      loss_type='sigmoid',
      data_collator=None,
      callbacks=None,
      optimizers=(None, None),
      preprocess_logits_for_metrics=None,
      peft_config=None,
      compute_metrics=None,
      special_stop_token='\n',
      dialogue_act_classifier=None,
      sample_frequency=1,
      hard_replacement_frequency=1,
  ):

    if beta is not None:
      args.beta = beta
    if label_smoothing is not None and hasattr(args, 'label_smoothing'):
      args.label_smoothing = label_smoothing
    if loss_type is not None:
      args.loss_type = [loss_type] if isinstance(loss_type, str) else loss_type

    # TRL 0.12+ renamed `tokenizer` -> `processing_class`.  Detect which one
    # the installed TRL version accepts so this code works on both the
    # training pod (TRL 0.11.x, uses `tokenizer`) and newer environments.
    _super_kwargs = dict(
        model=model,
        ref_model=ref_model,
        args=args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        optimizers=optimizers,
        peft_config=peft_config,
    )
    import inspect as _inspect_
    _params_ = _inspect_.signature(super().__init__).parameters
    if 'processing_class' in _params_:
      _super_kwargs['processing_class'] = tokenizer
    else:
      _super_kwargs['tokenizer'] = tokenizer
    super().__init__(**_super_kwargs)

    self.data_collator = _ACTMetadataCollator(
        self.data_collator,
        (
            'prompt',
            'chosen',
            'rejected',
            'chosen_policy',
            'rejected_policy',
            'gold_target',
            'gold_trajectory',
        ),
    )

    self.global_batches = 0
    self.is_in_evaluate = False
    self.tokenizer = tokenizer
    self.processing_class = tokenizer
    self.metrics = metrics

    self.special_stop_token = tokenizer.encode(special_stop_token)[-1]
    self.dialogue_act_classifier = dialogue_act_classifier
    self.sample_frequency = sample_frequency
    self.action_model = action_model
    self.user_simulator = user_simulator
    self.intent_summarization_model = intent_summarization_model
    self.hard_replacement_frequency = hard_replacement_frequency

    self.clarify_loss_weight = getattr(args, 'clarify_loss_weight', 1.0)

    self.mode_config = BetaDPOModeConfig(
        experiment_mode=args.experiment_mode,
        base_beta=args.beta,
        alpha=args.beta_dpo_alpha,
        filter_ratio=args.beta_dpo_filter_ratio,
        ema_gamma=args.beta_dpo_ema_gamma,
        min_beta=args.beta_dpo_min_beta,
        max_beta=getattr(args, 'beta_dpo_max_beta', 0.15),
        warmup_steps=getattr(args, 'beta_dpo_warmup_steps', 0),
    )
    self.mode_config.validate()

    _lt = getattr(self, 'loss_type', None) or getattr(self, 'loss_types', ['sigmoid'])
    loss_types = _lt if isinstance(_lt, list) else [_lt]
    _modes_allowing_non_sigmoid = {
        'act_original', 'act_ipo', 'act_simpo', 'act_orpo',
    }
    if self.mode_config.experiment_mode not in _modes_allowing_non_sigmoid and any(
        loss != 'sigmoid' for loss in loss_types
    ):
      raise ValueError(
          'UCP-Clarify beta-DPO modes currently support only sigmoid DPO loss.'
      )

    stats_device = self.accelerator.device
    self._running_gap_mean = torch.zeros(1, device=stats_device)
    self._running_gap_std = torch.zeros(1, device=stats_device)
    self._gap_stats_initialized = False

    self._cached_rollout_batch = None
    self._last_act_stats = self._empty_act_stats()
    self._act_rollout_totals = self._empty_act_stats()

  def _empty_act_stats(self):
    return {
        'wrong_action_loser_replacements': 0.0,
        'good_trajectory_winner_replacements': 0.0,
        'bad_trajectory_loser_replacements': 0.0,
    }

  def tokenize_row(
      self,
      features,
      processing_class,
      max_prompt_length,
      max_completion_length,
      add_special_tokens,
  ):
    tokenized = super().tokenize_row(
        features,
        processing_class,
        max_prompt_length,
        max_completion_length,
        add_special_tokens,
    )
    # Preserve UCP-Clarify metadata so the rollout step can update pairs on-policy.
    for key in (
        'prompt',
        'chosen',
        'rejected',
        'chosen_policy',
        'rejected_policy',
        'gold_target',
        'gold_trajectory',
    ):
      if key in features:
        tokenized[key] = features[key]
    return tokenized

  def evaluate(
      self,
      eval_dataset=None,
      ignore_keys=None,
      metric_key_prefix='eval',
  ):
    self.is_in_evaluate = True
    self._cached_rollout_batch = None
    self._last_act_stats = self._empty_act_stats()
    metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
    self.is_in_evaluate = False
    return metrics

  def post_process_features(self, features, metadata):
    if ' '.join(features['chosen'].lower().split()) == ' '.join(
        features['rejected'].lower().split()
    ):
      logging.info('Chosen is identical to rejected. Attempting to replace.')
      logging.info(
          'Replacing chosen %s -> %s and rejected %s -> %s',
          features['chosen'],
          metadata['chosen'],
          features['rejected'],
          metadata['rejected'],
      )
      features = {
          'prompt': features['prompt'],
          'chosen': metadata['chosen'],
          'rejected': metadata['rejected'],
      }
      if features['chosen'] == features['rejected']:
        rejected = 'Assistant: Could you clarify what you are asking for?'
        logging.info(
            'Still identical. rejected: %s -> %s',
            metadata['rejected'],
            rejected,
        )
        features['rejected'] = rejected
    elif features['chosen'].startswith('User:'):
      if 'Assistant:' in features['chosen']:
        features['chosen'] = (
            'Assistant: ' + features['chosen'].split('Assistant:')[-1].strip()
        )
      else:
        features['chosen'] = metadata['chosen']
      logging.info('Buggy rollout; New features: %s', features)
    return features

  def _act_tokenize_and_collate(self, features):
    tokenized = Dataset.from_dict(features)
    tokenized = tokenized.map(
        self.tokenize_row,
        fn_kwargs={
            'processing_class': self.processing_class,
            'max_prompt_length': self.max_prompt_length,
            'max_completion_length': getattr(
                self, 'max_completion_length', None
            ),
            'add_special_tokens': getattr(self, 'add_special_tokens', True),
        },
        load_from_cache_file=False,
    )
    tokenized_rows = [tokenized[i] for i in range(len(tokenized))]
    collated = self.data_collator(tokenized_rows)
    return {
        key: value.to(self.accelerator.device)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in collated.items()
    }

  def _prepare_rollout_batch(self, model, batch):
    os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
    self.global_batches += 1
    self._last_act_stats = self._empty_act_stats()

    should_rollout = (
        not self.is_in_evaluate
        and self.sample_frequency > 0
        and self.global_batches % self.sample_frequency == 0
    )
    if not should_rollout:
      self._cached_rollout_batch = batch
      return batch

    new_batch = defaultdict(list)
    for i, input_ids in enumerate(batch['prompt_input_ids']):
      simulator = Simulator(
          model=model,
          tokenizer=self.tokenizer,
          user_intent_model=self.intent_summarization_model,
          user_simulator_model=self.user_simulator,
          action_model=self.action_model,
      )
      trajectory = simulator.generate_trajectory(
          inputs=input_ids,
          chosen_policy=batch['chosen_policy'][i],
          prompt=batch['prompt'][i],
          gold_trajectory=batch['gold_trajectory'][i],
          max_input_length=self.max_length,
      )

      expected_action = None
      if 'chosen_policy' in batch:
        expected_action = self.action_model.mapper(batch['chosen_policy'][i])
      wrong_action = (
          expected_action is not None
          and trajectory['inferred_action'] != expected_action
      )
      trajectory_is_bad = self.metrics.conditon_checker(
          prompt=batch['prompt'][i],
          gold_target=batch['gold_target'][i],
          final_answer=trajectory['final_answer'],
          gold_trajectory=batch['gold_trajectory'][i],
          response=trajectory['response'],
      )

      if wrong_action:
        chosen = batch['chosen'][i]
        rejected = trajectory['response']
        self._last_act_stats['wrong_action_loser_replacements'] += 1
      elif trajectory_is_bad:
        chosen = batch['chosen'][i]
        rejected = trajectory['response']
        self._last_act_stats['bad_trajectory_loser_replacements'] += 1
      else:
        chosen = trajectory['response']
        rejected = batch['rejected'][i]
        self._last_act_stats['good_trajectory_winner_replacements'] += 1

      features = {
          'prompt': batch['prompt'][i],
          'chosen': chosen,
          'rejected': rejected,
      }
      features = self.post_process_features(
          features,
          {
              'chosen': batch['chosen'][i],
              'rejected': batch['rejected'][i],
          },
      )

      for key, value in features.items():
        new_batch[key].append(value)

      logging.info(
          (
              'UCP-Clarify rollout | expected_action=%s inferred_action=%s '
              'wrong_action=%s trajectory_is_bad=%s\nChosen:\n%s\nRejected:\n%s'
          ),
          expected_action,
          trajectory['inferred_action'],
          wrong_action,
          trajectory_is_bad,
          features['chosen'],
          features['rejected'],
      )

    for key, value in self._last_act_stats.items():
      self._act_rollout_totals[key] += value

    self._cached_rollout_batch = self._act_tokenize_and_collate(new_batch)
    return self._cached_rollout_batch

  def concatenated_forward(self, model, batch, is_ref_model=False):
    # trl >= 0.15 dropped the is_ref_model kwarg from the base
    # DPOTrainer.concatenated_forward signature AND from its own call sites
    # (e.g. compute_ref_log_probs). Infer it from model identity or an
    # explicit flag set by get_batch_loss_metrics before the ref pass.
    if not is_ref_model:
      ref = getattr(self, "ref_model", None)
      if ref is not None and model is ref:
        is_ref_model = True
      elif getattr(self, "_act_ref_forward_pending", False):
        is_ref_model = True

    if is_ref_model:
      rollout_batch = (
          self._cached_rollout_batch
          if self._cached_rollout_batch is not None
          else batch
      )
    else:
      rollout_batch = self._prepare_rollout_batch(model, batch)

    try:
      return super().concatenated_forward(
          model, rollout_batch, is_ref_model=is_ref_model
      )
    except TypeError:
      return super().concatenated_forward(model, rollout_batch)

  def _metric_mean(self, value):
    value = torch.atleast_1d(value.detach())
    return self.accelerator.gather_for_metrics(value).float().mean().item()

  def _metric_sum(self, value):
    value = torch.atleast_1d(value.detach())
    return self.accelerator.gather_for_metrics(value).float().sum().item()

  def _act_metrics(self, train_eval: Literal['train', 'eval']):
    prefix = 'eval_' if train_eval == 'eval' else ''
    device = self.accelerator.device
    metrics = {}
    for key, value in self._last_act_stats.items():
      metrics[f'{prefix}act/{key}'] = self._metric_sum(
          torch.tensor([value], device=device)
      )
    for key, value in self._act_rollout_totals.items():
      metrics[f'{prefix}act/total_{key}'] = self._metric_sum(
          torch.tensor([value], device=device)
      )
    return metrics

  def _compute_original_losses(
      self,
      model_output,
      ref_chosen_logps,
      ref_rejected_logps,
  ):
    _lt = getattr(self, 'loss_type', None) or getattr(self, 'loss_types', ['sigmoid'])
    loss_types = _lt if isinstance(_lt, list) else [_lt]
    losses = 0
    chosen_rewards = 0
    rejected_rewards = 0
    for idx, single_loss_type in enumerate(loss_types):
      # trl >= 0.15 changed DPOTrainer.dpo_loss signature: it now takes
      # (chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps)
      # and reads loss_type/aux_loss from self. Older trl took loss_type and
      # model_output as positional args. Handle both.
      try:
        _losses, _chosen_rewards, _rejected_rewards = self.dpo_loss(
            model_output['chosen_logps'],
            model_output['rejected_logps'],
            ref_chosen_logps,
            ref_rejected_logps,
            single_loss_type,
            model_output,
        )
      except TypeError:
        prev_loss_type = getattr(self, 'loss_type', None)
        try:
          self.loss_type = single_loss_type
          _losses, _chosen_rewards, _rejected_rewards = self.dpo_loss(
              model_output['chosen_logps'],
              model_output['rejected_logps'],
              ref_chosen_logps,
              ref_rejected_logps,
          )
        finally:
          if prev_loss_type is not None:
            self.loss_type = prev_loss_type
      loss_weights = getattr(self, 'loss_weights', None)
      weight = loss_weights[idx] if loss_weights else 1.0
      losses = losses + _losses * weight
      chosen_rewards = chosen_rewards + _chosen_rewards * weight
      rejected_rewards = rejected_rewards + _rejected_rewards * weight
    return losses, chosen_rewards, rejected_rewards

  def get_batch_loss_metrics(
      self,
      model,
      batch,
      train_eval: Literal['train', 'eval'] = 'train',
  ):
    metrics = {}
    model_output = self.concatenated_forward(model, batch)

    if 'ref_chosen_logps' in batch and 'ref_rejected_logps' in batch:
      ref_chosen_logps = batch['ref_chosen_logps']
      ref_rejected_logps = batch['ref_rejected_logps']
    else:
      self._act_ref_forward_pending = True
      try:
        ref_chosen_logps, ref_rejected_logps = self.compute_ref_log_probs(batch)
      finally:
        self._act_ref_forward_pending = False

    discrepancy = compute_discrepancy(
        model_output['chosen_logps'],
        model_output['rejected_logps'],
        ref_chosen_logps,
        ref_rejected_logps,
        reference_free=self.reference_free,
    )
    global_discrepancy = self.accelerator.gather(discrepancy.detach())
    (
        self._running_gap_mean,
        self._running_gap_std,
        self._gap_stats_initialized,
    ) = update_running_moments(
        global_discrepancy,
        self._running_gap_mean,
        self._running_gap_std,
        self.mode_config.ema_gamma,
        self._gap_stats_initialized,
    )

    if self.mode_config.experiment_mode == 'act_original':
      losses, chosen_rewards, rejected_rewards = self._compute_original_losses(
          model_output,
          ref_chosen_logps,
          ref_rejected_logps,
      )
      local_filter_weights = torch.ones_like(losses)
      beta_used = torch.full_like(losses, self.mode_config.base_beta)
      batch_std = global_discrepancy.std(unbiased=False)
      selected_mean = global_discrepancy.mean()
      actual_filter_ratio = torch.zeros(1, device=losses.device)
    else:
      protected_mask = None
      if self.mode_config.use_filtering and 'chosen_policy' in batch:
        protected_mask = torch.tensor(
            [p == 'CLARIFY' for p in batch['chosen_policy']],
            dtype=torch.bool,
            device=self.accelerator.device,
        )
      beta_dpo = compute_beta_dpo(
          config=self.mode_config,
          local_discrepancy=discrepancy,
          global_discrepancy=global_discrepancy,
          running_mean=self._running_gap_mean,
          running_std=self._running_gap_std,
          process_index=self.accelerator.process_index,
          num_processes=self.accelerator.num_processes,
          global_step=self.state.global_step,
          chosen_logps=model_output['chosen_logps'],
          rejected_logps=model_output['rejected_logps'],
          ref_chosen_logps=ref_chosen_logps,
          ref_rejected_logps=ref_rejected_logps,
          reference_free=self.reference_free,
          protected_mask=protected_mask,
      )
      losses = beta_dpo.losses
      chosen_rewards = beta_dpo.chosen_rewards
      rejected_rewards = beta_dpo.rejected_rewards
      local_filter_weights = beta_dpo.local_filter_weights
      beta_used = beta_dpo.beta_used
      batch_std = beta_dpo.batch_std
      selected_mean = beta_dpo.selected_mean
      actual_filter_ratio = beta_dpo.actual_filter_ratio

    reward_accuracies = (chosen_rewards > rejected_rewards).float()
    prefix = 'eval_' if train_eval == 'eval' else ''

    metrics[f'{prefix}rewards/chosen'] = self._metric_mean(chosen_rewards)
    metrics[f'{prefix}rewards/rejected'] = self._metric_mean(rejected_rewards)
    metrics[f'{prefix}rewards/accuracies'] = self._metric_mean(
        reward_accuracies
    )
    metrics[f'{prefix}rewards/margins'] = self._metric_mean(
        chosen_rewards - rejected_rewards
    )
    metrics[f'{prefix}logps/chosen'] = self._metric_mean(
        model_output['chosen_logps']
    )
    metrics[f'{prefix}logps/rejected'] = self._metric_mean(
        model_output['rejected_logps']
    )
    metrics[f'{prefix}logits/chosen'] = self._metric_mean(
        model_output['mean_chosen_logits']
    )
    metrics[f'{prefix}logits/rejected'] = self._metric_mean(
        model_output['mean_rejected_logits']
    )

    metrics.update(self._act_metrics(train_eval))
    metrics[f'{prefix}act/reward_discrepancy_mean'] = (
        global_discrepancy.mean().item()
    )
    metrics[f'{prefix}act/reward_discrepancy_std'] = batch_std.item()
    metrics[f'{prefix}act/running_discrepancy_mean'] = (
        self._running_gap_mean.item()
    )
    metrics[f'{prefix}act/running_discrepancy_std'] = (
        self._running_gap_std.item()
    )
    metrics[f'{prefix}act/discrepancy_selected_mean'] = selected_mean.item()
    metrics[f'{prefix}act/beta_batch'] = self._metric_mean(beta_used)
    metrics[f'{prefix}act/filter_ratio'] = actual_filter_ratio.item()

    if self.clarify_loss_weight != 1.0 and 'chosen_policy' in batch:
      class_weights = torch.ones_like(losses)
      for i, policy in enumerate(batch['chosen_policy']):
        if policy == 'CLARIFY':
          class_weights[i] = self.clarify_loss_weight
      losses = losses * class_weights

    weighted_losses = losses * local_filter_weights
    effective_weight = local_filter_weights.sum().clamp(min=1e-6)
    return weighted_losses.sum() / effective_weight, metrics

  def _compute_loss(self, model, inputs, return_outputs):
    """Override TRL 1.0.0's _compute_loss to inject UCP-Clarify custom DPO logic."""
    mode = 'train' if self.model.training else 'eval'
    device = self.accelerator.device

    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    completion_mask = inputs['completion_mask']
    model_kwargs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'use_cache': False,
    }
    for key in ('token_type_ids', 'pixel_values', 'pixel_attention_mask',
                'image_grid_thw', 'image_sizes'):
      if key in inputs:
        model_kwargs[key] = inputs[key]

    outputs = model(**model_kwargs)
    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    shift_completion_mask = completion_mask[..., 1:].contiguous()
    per_token_logps = selective_log_softmax(shift_logits, shift_labels)
    per_token_logps[shift_completion_mask == 0] = 0.0
    logps = per_token_logps.sum(dim=1)
    chosen_logps, rejected_logps = logps.chunk(2, dim=0)

    if self.precompute_ref_logps:
      ref_chosen_logps = inputs['ref_chosen_logps']
      ref_rejected_logps = inputs['ref_rejected_logps']
    else:
      with torch.no_grad(), disable_gradient_checkpointing(
          self.model, self.args.gradient_checkpointing_kwargs
      ):
        if _is_peft_model(model) and self.ref_model is None:
          unwrapped = self.accelerator.unwrap_model(model)
          adapter_name = 'ref' if 'ref' in unwrapped.peft_config else None
          if adapter_name:
            unwrapped.set_adapter(adapter_name)
            ref_outputs = self.model(**model_kwargs)
            unwrapped.set_adapter('default')
          else:
            with unwrapped.disable_adapter():
              ref_outputs = self.model(**model_kwargs)
        else:
          ref_outputs = self.ref_model(**model_kwargs)
      ref_shift_logits = ref_outputs.logits[..., :-1, :].contiguous()
      ref_per_token_logps = selective_log_softmax(ref_shift_logits, shift_labels)
      ref_per_token_logps[shift_completion_mask == 0] = 0.0
      ref_logps = ref_per_token_logps.sum(dim=1)
      ref_chosen_logps, ref_rejected_logps = ref_logps.chunk(2, dim=0)

    _ref_free = getattr(self, 'reference_free', False)
    discrepancy = compute_discrepancy(
        chosen_logps, rejected_logps,
        ref_chosen_logps, ref_rejected_logps,
        reference_free=_ref_free,
    )
    global_discrepancy = self.accelerator.gather(discrepancy.detach())
    (
        self._running_gap_mean,
        self._running_gap_std,
        self._gap_stats_initialized,
    ) = update_running_moments(
        global_discrepancy,
        self._running_gap_mean,
        self._running_gap_std,
        self.mode_config.ema_gamma,
        self._gap_stats_initialized,
    )

    if self.mode_config.experiment_mode in (
        'act_ipo', 'act_simpo', 'act_orpo'
    ):
      # Vanilla preference-optimization baselines reusing UCP-Clarify rollout and
      # clarify_loss_weight but swapping the pairwise loss.  Each branch is
      # self-contained and references only tensors already computed above.
      chosen_mask_p, rejected_mask_p = shift_completion_mask.chunk(2, dim=0)
      chosen_lens = chosen_mask_p.sum(dim=1).clamp(min=1).float()
      rejected_lens = rejected_mask_p.sum(dim=1).clamp(min=1).float()
      base_beta = self.mode_config.base_beta

      if self.mode_config.experiment_mode == 'act_ipo':
        # IPO (Azar et al. 2023): (Δlogπ − Δlogπ_ref − 1/(2β))^2.
        per_sequence_loss = (discrepancy - 1.0 / (2.0 * base_beta)) ** 2
      elif self.mode_config.experiment_mode == 'act_simpo':
        # SimPO (Meng et al. 2024): reference-free, length-normalized.
        simpo_beta = float(getattr(self.args, 'simpo_beta', 2.0))
        simpo_gamma = float(getattr(self.args, 'simpo_gamma', 1.0))
        avg_c = chosen_logps / chosen_lens
        avg_r = rejected_logps / rejected_lens
        per_sequence_loss = -F.logsigmoid(
            simpo_beta * (avg_c - avg_r) - simpo_gamma
        )
      else:  # act_orpo
        # ORPO (Hong et al. 2024): NLL(chosen) + λ · OR loss, reference-free.
        orpo_lambda = float(getattr(self.args, 'orpo_lambda', 0.1))
        avg_c = chosen_logps / chosen_lens
        avg_r = rejected_logps / rejected_lens
        x_c = avg_c.clamp(max=-1e-6)
        x_r = avg_r.clamp(max=-1e-6)
        log1mexp_c = torch.where(
            x_c < -0.693,
            torch.log1p(-torch.exp(x_c)),
            torch.log(-torch.expm1(x_c)),
        )
        log1mexp_r = torch.where(
            x_r < -0.693,
            torch.log1p(-torch.exp(x_r)),
            torch.log(-torch.expm1(x_r)),
        )
        log_odds = (avg_c - avg_r) - (log1mexp_c - log1mexp_r)
        or_loss = -F.logsigmoid(log_odds)
        nll_loss = -avg_c
        per_sequence_loss = nll_loss + orpo_lambda * or_loss

      local_filter_weights = torch.ones_like(per_sequence_loss)
      beta_used = torch.full_like(per_sequence_loss, base_beta)
      batch_std = global_discrepancy.std(unbiased=False)
      selected_mean = global_discrepancy.mean()
      actual_filter_ratio = torch.zeros(1, device=device)
    elif self.mode_config.experiment_mode == 'act_original':
      per_sequence_loss = -F.logsigmoid(
          self.mode_config.base_beta * discrepancy
      )
      local_filter_weights = torch.ones_like(per_sequence_loss)
      beta_used = torch.full_like(
          per_sequence_loss, self.mode_config.base_beta
      )
      batch_std = global_discrepancy.std(unbiased=False)
      selected_mean = global_discrepancy.mean()
      actual_filter_ratio = torch.zeros(1, device=device)
    else:
      protected_mask = None
      if self.mode_config.use_filtering and 'chosen_policy' in inputs:
        protected_mask = torch.tensor(
            [p == 'CLARIFY' for p in inputs['chosen_policy']],
            dtype=torch.bool, device=device,
        )
      beta_dpo = compute_beta_dpo(
          config=self.mode_config,
          local_discrepancy=discrepancy,
          global_discrepancy=global_discrepancy,
          running_mean=self._running_gap_mean,
          running_std=self._running_gap_std,
          process_index=self.accelerator.process_index,
          num_processes=self.accelerator.num_processes,
          global_step=self.state.global_step,
          chosen_logps=chosen_logps,
          rejected_logps=rejected_logps,
          ref_chosen_logps=ref_chosen_logps,
          ref_rejected_logps=ref_rejected_logps,
          reference_free=_ref_free,
          protected_mask=protected_mask,
      )
      per_sequence_loss = beta_dpo.losses
      local_filter_weights = beta_dpo.local_filter_weights
      beta_used = beta_dpo.beta_used
      batch_std = beta_dpo.batch_std
      selected_mean = beta_dpo.selected_mean
      actual_filter_ratio = beta_dpo.actual_filter_ratio

    chosen_rewards = (
        self.mode_config.base_beta
        * (chosen_logps - ref_chosen_logps)
    ).detach()
    rejected_rewards = (
        self.mode_config.base_beta
        * (rejected_logps - ref_rejected_logps)
    ).detach()

    if self.clarify_loss_weight != 1.0 and 'chosen_policy' in inputs:
      class_weights = torch.ones_like(per_sequence_loss)
      for i, policy in enumerate(inputs['chosen_policy']):
        if policy == 'CLARIFY':
          class_weights[i] = self.clarify_loss_weight
      per_sequence_loss = per_sequence_loss * class_weights

    weighted_losses = per_sequence_loss * local_filter_weights
    effective_weight = local_filter_weights.sum().clamp(min=1e-6)
    loss = weighted_losses.sum() / effective_weight

    per_token_entropy = entropy_from_logits(shift_logits.detach())
    entropy = per_token_entropy[shift_completion_mask.bool()].mean()
    entropy = self.accelerator.gather_for_metrics(entropy).mean().item()
    self._metrics[mode]['entropy'].append(entropy)

    if mode == 'train':
      num_tokens = self.accelerator.gather_for_metrics(
          inputs['attention_mask'].sum()
      ).sum().item()
      self._total_train_tokens += num_tokens
    self._metrics[mode]['num_tokens'] = [self._total_train_tokens]

    chosen_logits_d, rejected_logits_d = (
        shift_logits.detach().chunk(2, dim=0)
    )
    chosen_mask, rejected_mask = shift_completion_mask.chunk(2, dim=0)
    tc = chosen_logits_d[chosen_mask.bool()].mean(-1).sum()
    tr = rejected_logits_d[rejected_mask.bool()].mean(-1).sum()
    nc = chosen_mask.sum()
    nr = rejected_mask.sum()
    tc = self.accelerator.gather_for_metrics(tc).sum().item()
    nc = self.accelerator.gather_for_metrics(nc).sum().item()
    tr = self.accelerator.gather_for_metrics(tr).sum().item()
    nr = self.accelerator.gather_for_metrics(nr).sum().item()
    self._metrics[mode]['logits/chosen'].append(tc / nc if nc > 0 else 0.0)
    self._metrics[mode]['logits/rejected'].append(tr / nr if nr > 0 else 0.0)

    predictions = chosen_logits_d.argmax(dim=-1)
    cm = shift_completion_mask[:len(shift_completion_mask) // 2].bool()
    cl = shift_labels[:len(shift_labels) // 2]
    correct = (predictions == cl) & cm
    total_tok = self.accelerator.gather_for_metrics(cm.sum())
    correct_tok = self.accelerator.gather_for_metrics(correct.sum())
    ts = total_tok.sum()
    self._metrics[mode]['mean_token_accuracy'].append(
        (correct_tok.sum() / ts).item() if ts > 0 else 0.0
    )

    agg_cr = self.accelerator.gather(chosen_rewards)
    agg_rr = self.accelerator.gather(rejected_rewards)
    self._metrics[mode]['rewards/chosen'].append(agg_cr.mean().item())
    self._metrics[mode]['rewards/rejected'].append(agg_rr.mean().item())

    reward_acc = (chosen_rewards > rejected_rewards).float()
    agg_ra = self.accelerator.gather(reward_acc)
    self._metrics[mode]['rewards/accuracies'].append(agg_ra.mean().item())

    margins = chosen_rewards - rejected_rewards
    agg_m = self.accelerator.gather(margins)
    self._metrics[mode]['rewards/margins'].append(agg_m.mean().item())

    self._metrics[mode]['logps/chosen'].append(
        self.accelerator.gather(chosen_logps).mean().item()
    )
    self._metrics[mode]['logps/rejected'].append(
        self.accelerator.gather(rejected_logps).mean().item()
    )

    self._metrics[mode]['act/beta_batch'] = [beta_used.mean().item()]
    self._metrics[mode]['act/filter_ratio'] = [actual_filter_ratio.item()]
    self._metrics[mode]['act/reward_discrepancy_mean'] = [
        global_discrepancy.mean().item()
    ]
    self._metrics[mode]['act/reward_discrepancy_std'] = [batch_std.item()]

    return (loss, outputs) if return_outputs else loss
