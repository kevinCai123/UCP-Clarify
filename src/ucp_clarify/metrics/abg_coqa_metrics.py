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

"""Abg-CoQA answer-quality metric using SentenceBERT cosine similarity.

The original ACT paper (Chen et al.) evaluates Abg-CoQA with
SentenceBERT-based semantic similarity (Reimers & Gurevych, 2019; Risch
et al., 2021) because Abg-CoQA answers are free-form natural-language
sentences where lexical F1 (DROP F1) undercredits valid paraphrases.

`AbgCoqaMetrics` drops in wherever `PacificMetrics` is used:
  * `get_metrics(predicted, gold)` -> `(exact_match_like, similarity)`
    where `exact_match_like` is 1.0 when similarity >= threshold, else 0.
  * `conditon_checker(...)` returns True when `similarity < threshold`,
    matching the `f1 < threshold` contract of `PacificMetrics`.

The class loads the SBERT model lazily the first time it is used, so
importing this module is free of side-effects.
"""

from __future__ import annotations

import ast
import re
from typing import Any, List, Tuple

import numpy as np

from ucp_clarify.metrics.base_metrics import BaseMetrics


# Default embedding model. all-mpnet-base-v2 is the Reimers & Gurevych
# high-quality sentence embedding model (~420 MB, 768-dim). Can be
# overridden via constructor.
_DEFAULT_SBERT_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Default similarity threshold for `conditon_checker` and `exact_match`.
# Empirically ~0.65-0.75 corresponds to "semantically equivalent" for
# short QA answers. Matches the thresholds reported by Risch et al.
_DEFAULT_SIM_THRESHOLD = 0.7


def _strip_list_wrapper(text: Any) -> str:
  """Unwrap ['answer'] -> answer, since `convert_abg_coqa.py` wraps answers
  in a one-element list to mimic PACIFIC's answer shape. Falls back to
  str() when parsing fails.
  """
  if text is None:
    return ""
  if isinstance(text, (list, tuple)):
    return " ".join(str(t) for t in text).strip()
  s = str(text).strip()
  if s.startswith("[") and s.endswith("]"):
    try:
      parsed = ast.literal_eval(s)
      if isinstance(parsed, (list, tuple)):
        return " ".join(str(t) for t in parsed).strip()
      return str(parsed).strip()
    except (ValueError, SyntaxError):
      pass
  return s


class AbgCoqaMetrics(BaseMetrics):
  """Sentence-BERT cosine similarity for Abg-CoQA answer quality."""

  def __init__(
      self,
      model_name: str = _DEFAULT_SBERT_MODEL,
      sim_threshold: float = _DEFAULT_SIM_THRESHOLD,
      device: str | None = None,
  ):
    self.model_name = model_name
    self.sim_threshold = sim_threshold
    self._device = device
    self._model = None  # lazy

  @property
  def model(self):
    if self._model is None:
      # Import inside property so test imports and codebase-wide imports
      # don't require `sentence_transformers` to be installed.
      from sentence_transformers import SentenceTransformer  # type: ignore

      device = self._device
      if device is None:
        try:
          import torch  # type: ignore

          device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
          device = "cpu"
      self._model = SentenceTransformer(self.model_name, device=device)
    return self._model

  def get_metrics(
      self,
      predicted: Any,
      gold: Any,
  ) -> Tuple[float, float]:
    """Return (exact_match_like, semantic_similarity) in [0, 1].

    `exact_match_like` is 1.0 when similarity >= threshold, else 0.0 -
    mirrors the PACIFIC (exact_match, drop_f1) return contract so the
    `BaseEvaluator` can consume it without changes.
    """
    pred = _strip_list_wrapper(predicted)
    ref = _strip_list_wrapper(gold)
    if not pred or not ref:
      return 0.0, 0.0

    # `sentence_transformers.util.cos_sim` is a lightweight cosine-sim
    # helper; we normalize embeddings ourselves to stay numpy-only and
    # sidestep torch tensor requirements downstream.
    embs = self.model.encode([pred, ref], normalize_embeddings=True)
    emb_pred = np.asarray(embs[0])
    emb_ref = np.asarray(embs[1])
    sim = float(np.dot(emb_pred, emb_ref))
    # clamp numerical drift from near-parallel unit vectors.
    sim = max(-1.0, min(1.0, sim))
    # Map cosine similarity in [-1, 1] to a "quality score" in [0, 1].
    # Negative similarity is vanishingly rare for English sentences and
    # should still count as "bad", so we clamp at 0 rather than linearly
    # rescale, matching DROP F1's non-negativity.
    quality = max(0.0, sim)
    em = 1.0 if quality >= self.sim_threshold else 0.0
    return em, quality

  def conditon_checker(self, **metadata) -> bool:
    """`True` when the trajectory's final answer is semantically far
    from the gold answer (below threshold). Mirrors PacificMetrics."""
    _, sim = self.get_metrics(
        metadata.get("final_answer"), metadata.get("gold_target")
    )
    return sim < self.sim_threshold
