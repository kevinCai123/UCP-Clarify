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

"""Dataset-aware metric selection.

`run_act.py` and `evaluate.py` call `build_metrics(config)` to get the
right answer-quality metric:

* PacificMetrics  - DROP F1 (token-bag), for PACIFIC and any path that
  doesn't look like Abg-CoQA. This is the default, matching the
  pre-existing behavior so every current PACIFIC run is unchanged.
* AbgCoqaMetrics  - SentenceBERT cosine similarity, used when the
  config's data paths point at an Abg-CoQA-style dataset. Uses SBERT
  for free-form Abg-CoQA answers where lexical F1 undercredits valid
  paraphrases.

The selection is purely a function of the data-path substrings, so no
config-schema changes are required. A downstream caller may still pass
an explicit metric instance to override this.
"""

from __future__ import annotations

from typing import Tuple


# Substrings that identify an Abg-CoQA-style dataset. Additions here
# should stay specific to avoid accidentally matching PACIFIC paths.
_ABG_MARKERS = ("abg_coqa", "/abg_", "abg_v", "abgcoqa")


def _looks_like_abg(*paths: str | None) -> bool:
  for p in paths:
    if not p:
      continue
    low = p.lower()
    if any(m in low for m in _ABG_MARKERS):
      return True
  return False


def build_metrics(config) -> Tuple["object", str]:
  """Return (metrics_instance, quality_score_name) for the given config.

  `quality_score_name` drives the column labels in the eval report
  (e.g., "DROP F1" -> "Turn-Level DROP F1", "SBERT" -> "Turn-Level SBERT").
  """
  data_cfg = getattr(config, "data_config", None)
  candidate_paths = []
  if data_cfg is not None:
    for attr in (
        "train_path",
        "validation_path",
        "eval_data",
        "eval_result_output_path",
        "eval_sample_output_path",
    ):
      candidate_paths.append(getattr(data_cfg, attr, None))

  if _looks_like_abg(*candidate_paths):
    # Defer the SBERT import so PACIFIC runs never pull in
    # sentence-transformers.
    from ucp_clarify.metrics.abg_coqa_metrics import AbgCoqaMetrics

    return AbgCoqaMetrics(), "SBERT"

  from ucp_clarify.metrics.pacific_metrics import PacificMetrics

  return PacificMetrics(), "DROP F1"
