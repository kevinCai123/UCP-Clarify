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

from collections.abc import Sequence
import copy
import json
import logging
import os
import sys

from absl import app
from ucp_clarify.config.flags import FLAGS, initialize_flags
from ucp_clarify.config.utils import get_default_sft_config
from ucp_clarify.config.utils import get_sft_config_from_dict
from ucp_clarify.config.utils import get_sft_config_from_flags
from ucp_clarify.config.utils import get_sft_config_from_json_path
from ucp_clarify.data.utils import get_datasets_from_config
from ucp_clarify.models.utils import (
    get_checkpoint,
    initialize_env,
    load_sft_models,
)
from ucp_clarify.utils.artifact_utils import write_all_artifacts
from ucp_clarify.utils.storage_utils import (
    write_text,
)
import transformers
from transformers.trainer_callback import TrainerCallback
from trl import SFTTrainer
import wandb

wandb.init(mode="disabled")

logger = logging.getLogger(__name__)
initialize_flags(get_default_sft_config())

def main(argv):
  """Main function for learning via SFT."""
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  if FLAGS.config != "":
    sft_config = get_sft_config_from_json_path(FLAGS.config)
  else:
    sft_config = get_sft_config_from_flags(FLAGS)

  if FLAGS.resume_from_checkpoint:
    sft_config.training_config.resume_from_checkpoint = FLAGS.resume_from_checkpoint

  logging.info(sft_config.training_config)

  initialize_env()

  #######
  # Setup
  #######
  logging.basicConfig(
      format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
      datefmt="%Y-%m-%d %H:%M:%S",
      handlers=[logging.StreamHandler(sys.stdout)],
  )
  log_level = sft_config.training_config.get_process_log_level()
  logger.setLevel(log_level)
  transformers.utils.logging.set_verbosity(log_level)
  transformers.utils.logging.enable_default_handler()
  transformers.utils.logging.enable_explicit_format()

  # Log on each process the small summary:
  logger.info(f"Policy Model parameters {sft_config.policy_model_config}")
  logger.info(f"Data parameters {sft_config.data_config}")
  logger.info(f"Training/evaluation parameters {sft_config.training_config}")

  # Set seed for reproducibility
  transformers.set_seed(sft_config.training_config.seed)

  #####################################
  # Load tokenizer and process datasets
  #####################################
  # Truncate from left to ensure we don't lose labels in final turn
  sft_config.data_config.truncation_side = "left"

  train_dataset, dev_dataset = get_datasets_from_config(sft_config)

  train_dataset = train_dataset.rename_columns({
      "input_text": "prompt",
      "output_text": "completion",
  })
  dev_dataset = dev_dataset.rename_columns({
      "input_text": "prompt",
      "output_text": "completion",
  })

  # TRL >= 0.15 SFTTrainer auto-tokenizes every string column.  Drop metadata
  # columns (dialogue_policy, gold_trajectory, chosen/rejected, etc.) that the
  # trainer does not need so it only tokenizes prompt + completion.
  sft_keep_cols = {"prompt", "completion"}
  train_dataset = train_dataset.remove_columns(
      [c for c in train_dataset.column_names if c not in sft_keep_cols]
  )
  dev_dataset = dev_dataset.remove_columns(
      [c for c in dev_dataset.column_names if c not in sft_keep_cols]
  )

  models = load_sft_models(sft_config)

  policy_model = models[0]


  #########################
  # Instantiate SFT trainer
  #########################
  trainer_args = copy.deepcopy(sft_config.training_config)
  # TRL >= 0.15 SFTTrainer adds an internal `text` column after applying the
  # chat template.  If remove_unused_columns=False that raw string survives
  # into the data collator and crashes tensorisation.  Force it to True here;
  # UCP-Clarify training keeps its own value via run_act.py.
  trainer_args.remove_unused_columns = True
  trainer = SFTTrainer(
      policy_model.model,
      args=trainer_args,
      train_dataset=train_dataset,
      eval_dataset=dev_dataset,
      processing_class=policy_model.tokenizer,
      peft_config=None,
  )


  ###############
  # Training loop
  ###############
  checkpoint = None
  if sft_config.training_config.resume_from_checkpoint is not None:
    checkpoint = sft_config.training_config.resume_from_checkpoint

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
  if sft_config.training_config.do_eval:
    logger.info("*** Evaluate ***")
    eval_metrics = trainer.evaluate()
    eval_metrics["eval_samples"] = len(dev_dataset)
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
  else:
    eval_metrics = None

  if trainer.accelerator.is_main_process:
    write_all_artifacts(sft_config, trainer, metrics, eval_metrics)

  logger.info("*** Evaluate complete! ***")


if __name__ == "__main__":
  app.run(main)
