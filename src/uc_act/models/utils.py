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

import os
from pathlib import Path
from typing import Optional, Union
import warnings

from uc_act.config.base_config import BaseConfig
from uc_act.config.base_config import BaseInitializationConfig
from uc_act.config.model.hf_model_config import HFModelConfig
from uc_act.config.model.model_config import ModelConfig
from uc_act.models.action_classifier_model import ActionClassifierModel
from uc_act.models.generative_model import (
    GoogleGenerativeModel,
    is_openai_model_name,
)
from uc_act.models.hf_model import HFModel
from uc_act.models.intent_model import UserIntentModel
from uc_act.models.preference_model import RejectedSampleModel
from uc_act.models.simulator_model import SimulatorModel
from dotenv import load_dotenv
from peft import LoraConfig, PeftConfig
import torch
from transformers.trainer_utils import get_last_checkpoint


# I'm initializing this with the models that I've tested so far, but in
# principle there's no particular reason to limit it to these models.
_HF_MODELS = [
    'google/gemma-2-2b-it',
    'google/gemma-2-9b-it',
    'HuggingFaceH4/zephyr-7b-beta',
    'HuggingFaceH4/zephyr-7b-gemma-sft-v0.1',
    'meta-llama/Llama-3.2-3B-Instruct',
    'meta-llama/Llama-3.1-8B-Instruct',
    'mistralai/Mistral-7B-Instruct-v0.3',
    'mistralai/Mistral-Nemo-Instruct-2407',
    'Qwen/Qwen3.5-0.8B',
    'Qwen/Qwen2.5-0.5B-Instruct',
    'Qwen/Qwen2.5-1.5B-Instruct',
    'Qwen/Qwen2.5-3B-Instruct',
    'Qwen/Qwen3-0.6B',
    'Qwen/Qwen3-1.7B',
    'Qwen/Qwen3-4B-Instruct-2507',
]


# I'm initializing this with the models that I've tested so far, but in
# principle there's no particular reason to limit it to these models.
_GOOGLE_MODELS = [
    'gemini-ultra',
    'gemini-1.5-pro-001',
    'gemini-1.5-pro-002',
    'gemini-2.0-flash-001',
    'gemini-2.0-flash-exp',
    'gemini-2.5-flash',
    'gemini-2.5-flash-lite',
]
_OPENAI_MODELS = [
    'openai',
    'openai-default',
    'gpt-4o-mini',
    'gpt-4.1-mini',
    'gpt-4.1',
]


_CLASSES = {
    'action': ActionClassifierModel,
    'user_simulator': SimulatorModel,
    'preference': RejectedSampleModel,
    'intent': UserIntentModel,
}


_SUPPORTED_POLICY_MODELS = _HF_MODELS
_SUPPORTED_ACTION_MODELS = _GOOGLE_MODELS + _OPENAI_MODELS
_SUPPORTED_USER_SIMULATOR_MODELS = _GOOGLE_MODELS + _OPENAI_MODELS
_SUPPORTED_INTENT_MODELS = _GOOGLE_MODELS + _OPENAI_MODELS
_SUPPORTED_REF_MODELS = _HF_MODELS


def initialize_env() -> None:
    load_dotenv()
    google_api_key = os.getenv('GOOGLE_API_KEY')

    if google_api_key:
      try:
        import google.generativeai as genai
      except ImportError:
        genai = None
      if genai is not None:
        genai.configure(api_key=google_api_key)


def get_checkpoint(output_dir: str) -> Path | None:
  last_checkpoint = None
  if os.path.isdir(output_dir):
    last_checkpoint = get_last_checkpoint(output_dir)
  return last_checkpoint


def get_peft_config(model_config) -> PeftConfig | None:
  if model_config.use_peft is False:
    return None

  peft_config = LoraConfig(
      r=model_config.lora_r,
      lora_alpha=model_config.lora_alpha,
      lora_dropout=model_config.lora_dropout,
      bias='none',
      task_type='CAUSAL_LM',
      target_modules=model_config.lora_target_modules,
      modules_to_save=model_config.lora_modules_to_save,
  )

  return peft_config


def _check_supported_model_configs(
    policy_model: Optional[ModelConfig],
    ref_model: Optional[ModelConfig],
    action_model: ModelConfig,
    user_simulator: ModelConfig,
    intent_model: ModelConfig,
):
  """Check that the model configs are supported."""
  if policy_model and (
      policy_model.model_id not in _SUPPORTED_POLICY_MODELS
      and not policy_model.model_path
  ):
    raise ValueError(
        f'Policy model {policy_model.model_id} is not supported. Please use one'
        f' of the following models: {_SUPPORTED_POLICY_MODELS}.'
    )
  if ref_model and (
      ref_model.model_id not in _SUPPORTED_REF_MODELS
      and not ref_model.model_path
  ):
    raise ValueError(
        f'Ref model {ref_model.model_id} is not supported. Please use one'
        f' of the following models: {_SUPPORTED_REF_MODELS}.'
    )
  if action_model.model_id not in _SUPPORTED_ACTION_MODELS:
    if not is_openai_model_name(action_model.model_id):
      raise ValueError(
          f'Action model {action_model.model_id} is not supported. Please use'
          f' one of the following models: {_SUPPORTED_ACTION_MODELS}'
      )
  if user_simulator.model_id not in _SUPPORTED_USER_SIMULATOR_MODELS:
    if not is_openai_model_name(user_simulator.model_id):
      raise ValueError(
          f'User simulator model {user_simulator.model_id} is not supported.'
          ' Please use one of the following models:'
          f' {_SUPPORTED_USER_SIMULATOR_MODELS}'
      )
  if intent_model.model_id not in _SUPPORTED_INTENT_MODELS:
    if not is_openai_model_name(intent_model.model_id):
      raise ValueError(
          f'Intent model {intent_model.model_id} is not supported. Please use'
          f' one of the following models: {_SUPPORTED_INTENT_MODELS}'
      )
  return


def _route_and_load_model(
    config: Union[BaseConfig, BaseInitializationConfig],
    model_config: Union[ModelConfig, HFModel],
    model_type: str,
) -> Union[
    SimulatorModel,
    UserIntentModel,
    ActionClassifierModel,
    GoogleGenerativeModel,
    HFModel,
]:
  """Route the model to the appropriate loader."""
  if model_config.model_id in _HF_MODELS or model_config.model_path:
    return load_hf_model(config, model_config)
  elif (
      model_config.model_id in _GOOGLE_MODELS
      or model_config.model_id in _OPENAI_MODELS
      or is_openai_model_name(model_config.model_id)
  ):
    return load_google_model(config, model_config, model_type)
  else:
    raise ValueError(
        f'Model {model_config.model_id} is not supported. Please use'
        f' one of the following models: {_HF_MODELS}, {_GOOGLE_MODELS}, or'
        f' {_OPENAI_MODELS}'
    )


def load_google_model(
    config: BaseConfig, model_config: ModelConfig, model_type: str
) -> GoogleGenerativeModel:
  """Load a model from Vertex."""
  return _CLASSES[model_type](config, model_config)


def load_hf_model(
    config: Union[BaseConfig, BaseInitializationConfig],
    model_config: HFModelConfig,
):
  """Load a model from Hugging Face."""
  tokenizer, model = HFModel.construct_hf_model(config, model_config)
  return HFModel(config, tokenizer, model, model_config)


def load_models(
    config: BaseConfig,
    load_policy_from_checkpoint: bool = False,
    load_ref_model: bool = True,
):
  _check_supported_model_configs(
      None if load_policy_from_checkpoint else config.policy_model_config,
      None if load_policy_from_checkpoint else config.policy_model_config,
      config.action_model_config,
      config.user_simulator_config,
      config.intent_model_config,
  )
  policy_model = (
      load_hf_model(config, config.policy_model_config)
      if load_policy_from_checkpoint
      else _route_and_load_model(
          config, config.policy_model_config, model_type='policy'
      )
  )
  ref_model = None
  if load_ref_model:
    ref_model = (
        load_hf_model(config, config.policy_model_config)
        if load_policy_from_checkpoint
        else _route_and_load_model(
            config, config.policy_model_config, model_type='policy'
        )
    )
  action_model = _route_and_load_model(
      config, config.action_model_config, model_type='action'
  )
  user_simulator = _route_and_load_model(
      config, config.user_simulator_config, model_type='user_simulator'
  )
  intent_summarization_model = _route_and_load_model(
      config, config.intent_model_config, model_type='intent'
  )
  return (
      policy_model,
      ref_model,
      action_model,
      user_simulator,
      intent_summarization_model,
  )


def load_sft_models(config: BaseInitializationConfig, load_policy_from_checkpoint: bool = False):
  policy_model = (
      load_hf_model(config, config.policy_model_config)
      if load_policy_from_checkpoint
      else _route_and_load_model(
          config, config.policy_model_config, model_type='policy'
      )
  )
  return (
      policy_model,
  )
