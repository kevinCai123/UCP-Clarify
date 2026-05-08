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

from typing import Union
import re

from ucp_clarify.config.base_config import BaseConfig
from ucp_clarify.config.model.model_config import ModelConfig
from ucp_clarify.models.generative_model import GoogleGenerativeModel
import torch


class ActionClassifierModel(GoogleGenerativeModel):

  def __init__(
      self,
      config,
      model_config,
  ):
    super().__init__(config, model_config)
    self.set_actions()

  def clean_response(self, text):
    """Clean the response text to improve action classification accuracy.
    
    Removes list brackets, 'Assistant:' prefix, and other formatting that
    might confuse the Gemini classifier.
    """
    if text is None:
      return ""
    
    cleaned = text
    
    # Remove list brackets: ['...'] or ["..."]
    cleaned = re.sub(r"^\s*\[[\'\"](.*)[\'\"]]\s*$", r"\1", cleaned)
    # Also handle multi-item lists: ['item1', 'item2']
    cleaned = re.sub(r"^\s*\[[\'\"]", "", cleaned)
    cleaned = re.sub(r"[\'\"]\]\s*$", "", cleaned)
    
    # Remove 'Assistant:' prefix if present
    cleaned = re.sub(r"^Assistant:\s*", "", cleaned, flags=re.IGNORECASE)
    
    # Clean up any remaining quotes at start/end
    cleaned = cleaned.strip().strip("'\"").strip()
    
    return cleaned

  def prepend_icl_examples(self, inputs):
    # Clean the input before sending to Gemini
    cleaned_inputs = self.clean_response(inputs)
    
    icl_examples = """
    Assistant: {}
    Is the Assistant's response a clarifying question? Yes or No.""".format(
        cleaned_inputs
    )
    return icl_examples

  def set_actions(self):
    # TODO: This currently only supports two actions. Need to change to support
    # more.
    if hasattr(self.config.action_model_config, 'positive_action'):
      self.positive_action = self.config.action_model_config.positive_action
    else:
      self.positive_action = "CLARIFY"

    if hasattr(self.config.action_model_config, 'negative_action'):
      self.negative_action = self.config.action_model_config.negative_action
    else:
      self.negative_action = "ANSWER"

    if hasattr(self.config.action_model_config, 'positive_response'):
      self.positive_response = self.config.action_model_config.positive_response
    else:
      self.positive_response = "AskClarification"

    if hasattr(self.config.action_model_config, 'negative_response'):
      self.negative_response = self.config.action_model_config.negative_response
    else:
      self.negative_response = "DirectlyAnswerQuestion"

  def construct_generation_kwargs(self):
    self.generation_kwargs = {
      'temperature': 0.1,
      'top_p': 0.95,
      'top_k': 40,
      'max_output_tokens': 3,
      'stop_token': "\n",
      'candidate_count': 1
    }
    stop_token = self.generation_kwargs.pop('stop_token', "\n")
    if stop_token:
      self.generation_kwargs['stop_sequences'] = stop_token
    else:
      self.generation_kwargs['stop_sequences'] = []

  def generate(self, inputs, **generation_kwargs):
    response = super().generate(self.prepend_icl_examples(inputs),
                                **self.generation_kwargs)
    return response

  def is_likely_clarification(self, text):
    """Rule-based fallback to detect clarifying questions.
    
    Checks if the text looks like a clarifying question based on
    common patterns (question words + question mark).
    """
    if text is None:
      return False
    
    cleaned = self.clean_response(text).lower()
    
    # Must contain a question mark
    if "?" not in cleaned:
      return False
    
    # Common clarifying question patterns
    clarify_patterns = [
        r"which\s+(year|period|type|kind|one)",
        r"what\s+(year|period|type|kind|do you mean|are you)",
        r"what\s+.*\s+are you asking",
        r"could you\s+(specify|clarify|explain)",
        r"do you mean",
        r"are you asking about",
        r"which\s+.*\s+are you",
    ]
    
    for pattern in clarify_patterns:
      if re.search(pattern, cleaned):
        return True
    
    return False

  def classify(self, result, original_response=None):
    """Classify the action based on Gemini's response.
    
    Args:
      result: Gemini's Yes/No response
      original_response: The original model response (for rule-based fallback)
    """
    if "yes" in result.lower():
      return self.positive_response
    
    # Rule-based fallback: if Gemini said No but the response looks like
    # a clarifying question, override to AskClarification
    if original_response is not None and self.is_likely_clarification(original_response):
      return self.positive_response
    
    return self.negative_response

  def mapper(self, policy):
    if (policy == self.positive_action) or (self.positive_action in policy):
      return "AskClarification"
    elif (policy == self.negative_action) or (self.negative_action in policy):
      return "DirectlyAnswerQuestion"
    else:
      print("Policy was odd: ", policy)
      return policy
