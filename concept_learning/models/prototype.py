"""Prototype model of category learning.

Theory (Rosch 1975; Posner & Keele 1968): a category is represented by a single
summary point -- the central tendency of its members -- and classification is by
similarity to that prototype. Nothing about individual instances is retained.

Implementation
--------------
* One prototype per category, maintained as the **running mean** of the feature
  vectors of the stimuli that feedback has assigned to that category. Early
  prototypes are noisy and shift as more instances arrive, giving a learning
  curve; they stabilise as counts grow.
* Classification uses an exponential similarity (city-block distance, sensitivity
  ``c``) to each prototype, combined by a Luce choice rule.

Predicted failure (a feature, not a bug)
----------------------------------------
For Types II and VI the two categories have *identical* means -- (0.5, 0.5, 0.5)
-- so the prototypes converge to the same point and the model is pinned at chance
forever. A prototype theory simply cannot represent XOR/parity. The
horizon-bounded runner reports this as "criterion not reached" rather than
looping.
"""

from __future__ import annotations

import numpy as np

from ..stimuli import N_DIMENSIONS, feature_matrix
from .base import CategoryModel


class PrototypeModel(CategoryModel):
    name = "Prototype"

    def __init__(self, sensitivity: float = 2.0):
        #: Specificity of the similarity gradient (higher = sharper).
        self.c = float(sensitivity)
        self._features = feature_matrix("01")  # 8 x 3, values in {0, 1}
        self.reset(np.random.default_rng())

    def reset(self, rng: np.random.Generator) -> None:
        # Running sum of feature vectors and counts, per category (0=A, 1=B).
        self._sum = np.zeros((2, N_DIMENSIONS), dtype=float)
        self._count = np.zeros(2, dtype=float)

    def _prototype(self, category: int) -> np.ndarray | None:
        if self._count[category] == 0:
            return None
        return self._sum[category] / self._count[category]

    def _similarity(self, x: np.ndarray, category: int) -> float:
        proto = self._prototype(category)
        if proto is None:
            return 0.0
        dist = np.abs(x - proto).sum()  # city-block
        return float(np.exp(-self.c * dist))

    def p_a(self, stim_idx: int) -> float:
        x = self._features[stim_idx]
        s_a = self._similarity(x, 0)
        s_b = self._similarity(x, 1)
        total = s_a + s_b
        if total == 0.0:      # no feedback yet -> guess
            return 0.5
        return s_a / total

    def update(self, stim_idx: int, response: int, true_label: int,
               rng: np.random.Generator) -> None:
        # Supervised: fold the instance into its true category's running mean.
        self._sum[true_label] += self._features[stim_idx]
        self._count[true_label] += 1.0
