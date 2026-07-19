"""Exemplar model of category learning: ALCOVE (Kruschke 1992).

Theory (Medin & Schaffer 1978; Nosofsky 1986): a category is the set of
remembered instances; classification is judged by attention-weighted similarity
to those stored exemplars. This is the Generalized Context Model (GCM).

ALCOVE keeps the GCM's classification core -- exemplar nodes, attention-weighted
city-block distance, exponential similarity, Luce choice -- and adds the
error-driven learning the errors-to-criterion paradigm requires. It is exactly
the model Nosofsky, Gluck, Palmeri, McKinley & Glauthier (1994) used to fit the
SHJ learning curves. A plain fixed-attention GCM has no real learning dynamics
here (once all 8 items are stored it is at asymptote), so ALCOVE is the right
exemplar learner.

The attention learning rate ``lambda_a`` is a **toggle**:

* ``lambda_a > 0`` -- full ALCOVE. Learns to attend to the relevant dimensions,
  which is what makes Type II (XOR of two dims, third irrelevant) *easy*.
* ``lambda_a == 0`` -- attention frozen uniform. A genuine learner (association
  weights still adapt) but without selective attention; the Type II advantage
  should collapse. This is the "exemplar theory without attention" baseline.

Architecture (per Kruschke 1992, with r = q = 1)
------------------------------------------------
* Hidden (exemplar) node j sits at each of the 8 stimulus positions psi_j.
* Hidden activation for input x:  h_j = exp(-c * sum_k alpha_k |psi_jk - x_k|)
* Output (category) activation:   out_o = sum_j w[o, j] h_j
* Response:  P(A) = softmax over {out_A, out_B} scaled by phi (Luce choice)
* Humble teacher:  t_o = max(out_o, 1) if o is correct else min(out_o, -1)
* Weight update:      w[o, j] += lambda_w (t_o - out_o) h_j
* Attention update:   alpha_k -= lambda_a * sum_j d_j h_j c |psi_jk - x_k|,
                      with d_j = sum_o (t_o - out_o) w[o, j],  alpha_k clipped >= 0
"""

from __future__ import annotations

import numpy as np

from ..stimuli import N_DIMENSIONS, N_STIMULI, feature_matrix
from .base import CategoryModel


class ALCOVEModel(CategoryModel):
    def __init__(
        self,
        c: float = 6.5,
        phi: float = 2.0,
        lambda_w: float = 0.03,
        lambda_a: float = 0.0033,
    ):
        # Defaults are Kruschke's (1992) fitted values for the SHJ data; the
        # question here is qualitative ordering, so we do not re-fit them.
        self.c = float(c)
        self.phi = float(phi)
        self.lambda_w = float(lambda_w)
        self.lambda_a = float(lambda_a)
        # Exemplar node positions = the 8 stimuli (fixed).
        self._psi = feature_matrix("01")                     # 8 x 3
        self.reset(np.random.default_rng())

    @property
    def name(self) -> str:
        return "Exemplar (ALCOVE)" if self.lambda_a > 0 else "Exemplar (ALCOVE, no attn)"

    def reset(self, rng: np.random.Generator) -> None:
        # Uniform initial attention; zero association weights.
        self._alpha = np.full(N_DIMENSIONS, 1.0 / N_DIMENSIONS, dtype=float)
        self._w = np.zeros((2, N_STIMULI), dtype=float)      # [category, exemplar]

    # -- forward pass -------------------------------------------------------
    def _hidden(self, x: np.ndarray) -> np.ndarray:
        dist = (self._alpha * np.abs(self._psi - x)).sum(axis=1)   # 8
        return np.exp(-self.c * dist)

    def _outputs(self, h: np.ndarray) -> np.ndarray:
        return self._w @ h                                          # 2

    def p_a(self, stim_idx: int) -> float:
        h = self._hidden(self._psi[stim_idx])
        out = self._outputs(h)
        # Luce / softmax choice scaled by phi, numerically stabilised.
        z = self.phi * out
        z -= z.max()
        e = np.exp(z)
        return float(e[0] / e.sum())

    # -- learning -----------------------------------------------------------
    def update(self, stim_idx: int, response: int, true_label: int,
               rng: np.random.Generator) -> None:
        x = self._psi[stim_idx]
        h = self._hidden(x)
        out = self._outputs(h)

        # Humble teacher targets.
        teacher = np.empty(2)
        for o in range(2):
            if o == true_label:
                teacher[o] = max(out[o], 1.0)
            else:
                teacher[o] = min(out[o], -1.0)
        err = teacher - out                                        # 2

        # Attention gradient uses the pre-update weights (same forward pass),
        # so compute it before the weight update. Skipped when lambda_a == 0.
        if self.lambda_a > 0.0:
            d = err @ self._w                                      # 8, per exemplar
            grad = (d * h) @ (self.c * np.abs(self._psi - x))      # 3, per dimension
            self._alpha -= self.lambda_a * grad
            np.clip(self._alpha, 0.0, None, out=self._alpha)

        # Association-weight update.
        self._w += self.lambda_w * np.outer(err, h)
