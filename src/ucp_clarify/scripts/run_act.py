# coding=utf-8
# Copyright 2025 The Google Research Authors.
# Modifications Copyright 2025 Songlin Cai.
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

from collections.abc import Sequence
import copy
import json
import logging
import os
import sys

from absl import app
import transformers
from transformers.trainer_callback import TrainerCallback
import wandb
from peft import LoraConfig

wandb.init(mode="disabled")

from ucp_clarify.config.base_config import BaseConfig
from ucp_clarify.config.flags import initialize_flags, FLAGS
from ucp_clarify.config.training.training_config import ACTConfig
from ucp_clarify.config.utils import (
    get_config_from_dict,
    get_config_from_flags,
    get_config_from_json_path,
    get_default_config,
)
from ucp_clarify.data.utils import get_datasets_from_config
from ucp_clarify.metrics.dispatch import build_metrics
from ucp_clarify.models.utils import (
    initialize_env,
    get_checkpoint,
    load_models,
)
from ucp_clarify.simulation.simulator import Simulator
from ucp_clarify.trainer import act_trainer
from ucp_clarify.utils.storage_utils import (
    write_text,
)
from ucp_clarify.utils.artifact_utils import write_all_artifacts

logger = logging.getLogger(__name__)
initialize_flags(get_default_config())

def main(argv):
  """Main function for the ACT algorithm."""
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  if FLAGS.config != "":
    act_config = get_config_from_json_path(FLAGS.config)
  else:
    act_config = get_config_from_flags(FLAGS)

  if FLAGS.resume_from_checkpoint:
    act_config.training_config.resume_from_checkpoint = FLAGS.resume_from_checkpoint

  logging.info(act_config.training_config)

  initialize_env()

  #######
  # Setup
  #######
  logging.basicConfig(
      format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
      datefmt="%Y-%m-%d %H:%M:%S",
      handlers=[logging.StreamHandler(sys.stdout)],
  )
  log_level = act_config.training_config.get_process_log_level()
  logger.setLevel(log_level)
  transformers.utils.logging.set_verbosity(log_level)
  transformers.utils.logging.enable_default_handler()
  transformers.utils.logging.enable_explicit_format()

  # Log on each process the small summary:
  logger.info(f"Policy Model parameters {act_config.policy_model_config}")
  logger.info(f"Data parameters {act_config.data_config}")
  logger.info(f"Training/evaluation parameters {act_config.training_config}")

  # Set seed for reproducibility
  transformers.set_seed(act_config.training_config.seed)

  #####################################
  # Load tokenizer and process datasets
  #####################################
  # Truncate from left to ensure we don't lose labels in final turn
  act_config.data_config.truncation_side = "left"

  train_dataset, dev_dataset = get_datasets_from_config(act_config)

  train_dataset = train_dataset.rename_columns({"input_text": "prompt"})
  dev_dataset = dev_dataset.rename_columns({"input_text": "prompt"})

  models = load_models(act_config, load_ref_model=False)

  policy_model, _, action_model, user_simulator, intent_summarization_model = (
      models
  )
  model = policy_model
  # Always use LoRA adapters. When reference_free=False, TRL uses the frozen
  # base model (LoRA disabled via null_ref_context) as the implicit reference,
  # giving standard DPO discrepancy without doubling GPU memory.
  ref_model = None
  peft_config = LoraConfig(
      r=8,
      lora_alpha=16,
      lora_dropout=0.05,
      bias='none',
      task_type='CAUSAL_LM',
      target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
  )


  #########################
  # Instantiate ACT trainer
  #########################


  trainer_args = copy.deepcopy(act_config.training_config)
  # Dataset-aware metric (PACIFIC -> DROP F1, Abg-CoQA -> SBERT). See
  # ucp_clarify/metrics/dispatch.py. `conditon_checker` on the selected metric
  # drives the ACT rollout re-labeling, so the training signal matches
  # the downstream eval metric on each dataset.
  metrics_obj, quality_score_name = build_metrics(act_config)
  logger.info(
      "Using answer-quality metric: %s (score name: %s)",
      type(metrics_obj).__name__,
      quality_score_name,
  )
  trainer = act_trainer.ACTTrainer(
      model.model,
      ref_model.model if ref_model is not None else None,
      args=trainer_args,
      beta=trainer_args.beta,
      train_dataset=train_dataset,
      eval_dataset=dev_dataset,
      tokenizer=policy_model.tokenizer,
      action_model=action_model,
      user_simulator=user_simulator,
      intent_summarization_model=intent_summarization_model,
      peft_config=peft_config,
      sample_frequency=trainer_args.sample_frequency,
      hard_replacement_frequency=-1,
      metrics=metrics_obj,
  )


  ###############
  # Training loop
  ###############
  checkpoint = None
  if act_config.training_config.resume_from_checkpoint is not None:
    checkpoint = act_config.training_config.resume_from_checkpoint

  train_result = trainer.train(resume_from_checkpoint=checkpoint)
  metrics = train_result.metrics
  metrics["train_samples"] = len(train_dataset)
  trainer.log_metrics("train", metrics)
  trainer.save_metrics("train", metrics)
  trainer.save_state()

  logger.info("*** Training complete ***")

  ##########
  # Evaluate
  ##########
  if act_config.training_config.do_eval:
    logger.info("*** Evaluate ***")
    eval_metrics = trainer.evaluate()
    eval_metrics["eval_samples"] = len(dev_dataset)
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
  else:
    eval_metrics = None

  if trainer.accelerator.is_main_process:
    write_all_artifacts(act_config, trainer, metrics, eval_metrics)

  logger.info("*** Evaluate complete! ***")


if __name__ == "__main__":
  app.run(main)
