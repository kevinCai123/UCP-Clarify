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

import io
import json
import os
import re
import subprocess
import tempfile
from typing import Any, Callable, Optional, Union
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
)



def save_model_to_file(path, trainer):
  """Save a trainer to a local file."""
  trainer.save_model(path)


def save_model(path, trainer):
    return save_model_to_file(path, trainer)




def load_model_from_file(
    path,
    base_model_class = AutoModelForCausalLM,
    base_tokenizer_class = None,
    **model_kwargs,
):
  """Load a saved training and optionally a tokenizer from a local file.

  Automatically detects PEFT/LoRA adapters (adapter_config.json present)
  and loads the base model first, then applies the adapter.
  """
  adapter_config_path = os.path.join(path, "adapter_config.json")
  if os.path.exists(adapter_config_path):
    from peft import PeftModel
    with open(adapter_config_path, "r") as f:
      adapter_cfg = json.load(f)
    base_path = adapter_cfg.get("base_model_name_or_path", None)
    if base_path is None:
      raise ValueError(f"adapter_config.json at {path} missing base_model_name_or_path")
    print(f"Loading PEFT adapter: base={base_path}, adapter={path}")
    base_model = base_model_class.from_pretrained(base_path, **model_kwargs)
    model = PeftModel.from_pretrained(base_model, path)
    model = model.merge_and_unload()
    if base_tokenizer_class is not None:
      tokenizer = base_tokenizer_class.from_pretrained(base_path)
      return model, tokenizer
    else:
      return model

  model = base_model_class.from_pretrained(path, **model_kwargs)
  if base_tokenizer_class is not None:
    tokenizer = base_tokenizer_class.from_pretrained(path)
    return model, tokenizer
  else:
    return model


def load_model(
    path,
    base_model_class = AutoModelForCausalLM,
    base_tokenizer_class = None,
    **model_kwargs,
):
    return load_model_from_file(
        path, base_model_class, base_tokenizer_class, **model_kwargs
    )




def read_file_from_file(path):
  """Reads a file from a local file."""
  with open(path, "r") as f:
    return f.read()


def read_file(path):
    return read_file_from_file(path)




def read_jsonl_from_file(path):
  """Reads a jsonl object from a file."""
  data = []
  with open(path, "r", encoding="utf-8") as f:
    for row in f:
      data.append(json.loads(row))
  return data


def read_jsonl(path):
    return read_jsonl_from_file(path)




def read_json_from_file(path):
  """Reads a json object from a local file."""
  print(f"Reading from {path}")
  with open(path, "r") as f:
    data = json.load(f)
  return data


def read_json(path):
    return read_json_from_file(path)




def write_text_to_file(path, contents):
  with open(path, "w") as f:
    f.write(contents)


def write_text(path, contents):
    write_text_to_file(path, contents)




def write_jsonl_to_file(path, content):
  """Writes a jsonl object to a file."""
  print(f"Writing to {path}")
  with open(path, "w", encoding="utf-8") as f:
    for m in content:
      f.write(json.dumps(m))
      f.write("\n")


def write_jsonl(path, content):
    write_jsonl_to_file(path, content)




def write_json_to_file(path, content):
  """Writes a JSON object to a local file."""
  print(f"Writing to {path}")
  with open(path, "w") as f:
    f.write(json.dumps(content, default=str))


def write_json(path, content):
    write_json_to_file(path, content)


