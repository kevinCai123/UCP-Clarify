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

import json
import logging
import os
import time
from typing import Optional, Union
import uuid

from uc_act.config.base_config import BaseConfig
from uc_act.config.model.model_config import ModelConfig
from uc_act.models.base_model import BaseModel
import torch
from openai import OpenAI


logger = logging.getLogger(__name__)


def is_openai_model_name(model_name: Optional[str]) -> bool:
  if model_name is None:
    return False
  lowered = model_name.lower()
  return lowered in {'openai', 'openai-default'} or lowered.startswith(
      ('gpt-', 'o1', 'o3', 'o4')
  )


def resolve_openai_model_name(model_name: Optional[str]) -> str:
  if model_name and model_name.lower() not in {'openai', 'openai-default'}:
    return model_name

  env_model = os.getenv('OPENAI_MODEL')
  if env_model:
    return env_model
  raise ValueError(
      'OPENAI_MODEL environment variable not set and no explicit OpenAI model '
      'id was provided.'
  )


class GoogleGenerativeModel(BaseModel):

  def __init__(self, config: BaseConfig, model_config: ModelConfig):
    super().__init__(config)
    self.model_config = model_config
    self.provider = (
        'openai'
        if is_openai_model_name(self.model_config.model_id)
        else 'google'
    )
    self.model_name = (
        resolve_openai_model_name(self.model_config.model_id)
        if self.provider == 'openai'
        else self.model_config.model_id
    )
    self.model = GoogleGenerativeModel.load_generative_model(
        self.model_name, provider=self.provider
    )
    self.construct_generation_kwargs()

  def construct_generation_kwargs(self):
    self.generation_kwargs = {
      'temperature': 0.2,
      'top_p': 0.92,
      'top_k': 40,
      'max_output_tokens': 64,
      'stop_token': "\n",
      'candidate_count': 1
    }
    stop_token = self.generation_kwargs.pop('stop_token', "\n")
    if stop_token:
      self.generation_kwargs['stop_sequences'] = stop_token
    else:
      self.generation_kwargs['stop_sequences'] = []

  def generate_batch(self, inputs: list[str], **generation_kwargs):
      raise NotImplementedError("Batch prediction not yet supported.")

  def generate(self, inputs: Union[str, torch.Tensor], **generation_kwargs):
    sleep_attempts = 0
    sleep_time_seconds = 2
    while sleep_attempts <= 5:
      try:
        if self.provider == 'openai':
          response = self.model.chat.completions.create(
              model=self.model_name,
              messages=[{'role': 'user', 'content': str(inputs)}],
              temperature=generation_kwargs.get('temperature'),
              top_p=generation_kwargs.get('top_p'),
              max_completion_tokens=generation_kwargs.get(
                  'max_output_tokens'
              ),
              stop=generation_kwargs.get('stop_sequences'),
          )
          message = response.choices[0].message
          return message.content or ''

        response = self.model.generate_content(
            contents=inputs,
            generation_config=generation_kwargs,
            safety_settings={
                self._harm_category.HARM_CATEGORY_HATE_SPEECH: (
                    self._harm_block_threshold.BLOCK_NONE
                ),
                self._harm_category.HARM_CATEGORY_DANGEROUS_CONTENT: (
                    self._harm_block_threshold.BLOCK_NONE
                ),
                self._harm_category.HARM_CATEGORY_HARASSMENT: (
                    self._harm_block_threshold.BLOCK_NONE
                ),
                self._harm_category.HARM_CATEGORY_SEXUALLY_EXPLICIT: (
                    self._harm_block_threshold.BLOCK_NONE
                ),
            },
        )
        response = response.candidates[0].content
        if len(response.parts) > 0:
          return response.parts[0].text
        else:
          return ''
      except Exception as re:
        print(f"Exception occurred while processing property: {re}")
        sleep_attempts += 1
        time.sleep(sleep_time_seconds)
        sleep_time_seconds *= 2

    return 'Error occurred while processing the request'

  @staticmethod
  def load_generative_model(model_name, provider='google'):
    if provider == 'openai':
      api_key = os.getenv('OPENAI_API_KEY')
      if not api_key:
        raise ValueError('OPENAI_API_KEY environment variable not set.')
      return OpenAI(api_key=api_key)

    try:
      from google.generativeai import GenerativeModel
      from google.generativeai.types import (
          HarmBlockThreshold,
          HarmCategory,
      )
    except ImportError as exc:
      raise ImportError(
          'google-generativeai is not installed, but a Gemini model was '
          'requested.'
      ) from exc

    GoogleGenerativeModel._harm_block_threshold = HarmBlockThreshold
    GoogleGenerativeModel._harm_category = HarmCategory
    return GenerativeModel(model_name)

