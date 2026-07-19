"""Rule-based model of category learning (v1: simple rules).

Theory (the classical view; Bruner, Goodnow & Austin 1956): a concept is an
explicit logical condition on features. Learning is a *search* through a space of
candidate rules -- form a hypothesis, keep it while it works, abandon it on error
(win-stay / lose-shift), preferring the simplest rule consistent with experience.
This follows Thagard's account of rules as IF-THEN conditions acquired by
inductive generalisation, and the RULEX tradition (Nosofsky, Palmeri & McKinley
1994) of trying simple rules first.

Hypothesis space (v1)
---------------------
Ordered by complexity, the model prefers the simplest consistent rule:

1. **Unidimensional** rules: "category A iff dimension d has value v"
   (3 dims x 2 polarities = 6 rules).
2. **Conjunctive** rules over two dimensions: "A iff (d1 = v1) AND (d2 = v2)",
   and the complementary mapping, covering the 2-literal conjunctions.

This space deliberately excludes XOR and parity. So the rule learner solves
Type I readily, but **Type II (XOR) has no consistent rule in the space** -- it
predicts Type II should be *hard*, the classic point of contrast with humans,
who find it easy. Types IV/VI likewise have no simple consistent rule. Adding
stored exceptions (RULEX-style) is the planned v2 extension; v1 isolates what a
pure simple-rule learner can and cannot do.

Because the model can tell whether its own rule erred, learning is driven by the
response/feedback comparison (lose-shift), not by a supervised target.
"""

from __future__ import annotations

from itertools import product

import numpy as np

from ..stimuli import N_DIMENSIONS, STIMULI
from .base import CategoryModel


def _enumerate_hypotheses() -> list[tuple[int, np.ndarray]]:
    """Return (complexity, label_vector) for every rule in the v1 space.

    A hypothesis is a length-8 vector of predicted labels (0/1) over STIMULI,
    tagged with a complexity (1 = unidimensional, 2 = conjunctive) so the model
    can prefer simpler rules. Duplicate label patterns keep their lowest
    complexity.
    """
    feats = np.array(STIMULI)                      # 8 x 3, {0,1}
    seen: dict[bytes, int] = {}
    hyps: list[tuple[int, np.ndarray]] = []

    def add(complexity: int, positive_mask: np.ndarray) -> None:
        # positive_mask marks stimuli assigned to category A (label 0).
        labels = np.where(positive_mask, 0, 1).astype(int)
        for lab in (labels, 1 - labels):           # A and B are interchangeable
            key = lab.tobytes()
            if key not in seen or complexity < seen[key]:
                seen[key] = complexity
        hyps.append((complexity, labels))

    # 1. Unidimensional: A iff dim d == v
    for d in range(N_DIMENSIONS):
        for v in (0, 1):
            add(1, feats[:, d] == v)

    # 2. Conjunctive: A iff (d1 == v1) AND (d2 == v2)
    for d1, d2 in ((0, 1), (0, 2), (1, 2)):
        for v1, v2 in product((0, 1), repeat=2):
            add(2, (feats[:, d1] == v1) & (feats[:, d2] == v2))

    # De-duplicate by label pattern, keeping the lowest complexity.
    best: dict[bytes, tuple[int, np.ndarray]] = {}
    for complexity, labels in hyps:
        key = labels.tobytes()
        if key not in best or complexity < best[key][0]:
            best[key] = (complexity, labels)
    return sorted(best.values(), key=lambda t: t[0])


#: The full v1 hypothesis space, computed once.
_HYPOTHESES: list[tuple[int, np.ndarray]] = _enumerate_hypotheses()


class RuleModel(CategoryModel):
    name = "Rule (v1: simple)"

    def __init__(self, lapse: float = 0.0):
        #: Small guessing rate on the current rule's prediction (0 = deterministic).
        self.lapse = float(lapse)
        self._hyps = _HYPOTHESES
        self.reset(np.random.default_rng())

    def reset(self, rng: np.random.Generator) -> None:
        # Observed labels so far (-1 = unseen), and a randomly chosen start rule.
        self._observed = np.full(len(STIMULI), -1, dtype=int)
        self._current = int(rng.integers(len(self._hyps)))

    def p_a(self, stim_idx: int) -> float:
        pred = self._hyps[self._current][1][stim_idx]     # 0 or 1
        p_a = 1.0 if pred == 0 else 0.0
        # Optional lapse: with prob `lapse`, guess uniformly.
        return (1.0 - self.lapse) * p_a + self.lapse * 0.5

    def _consistency(self, labels: np.ndarray) -> int:
        """How many *seen* stimuli this rule labels correctly."""
        seen = self._observed >= 0
        if not seen.any():
            return 0
        return int((labels[seen] == self._observed[seen]).sum())

    def update(self, stim_idx: int, response: int, true_label: int,
               rng: np.random.Generator) -> None:
        self._observed[stim_idx] = true_label

        # Win-stay: if the current rule is still consistent with everything
        # seen, keep it.
        current_labels = self._hyps[self._current][1]
        seen = self._observed >= 0
        if (current_labels[seen] == self._observed[seen]).all():
            return

        # Lose-shift: pick the simplest rule with maximal consistency; break
        # ties at random. This prefers unidimensional rules, then conjunctions.
        best_score = None
        candidates: list[int] = []
        for i, (complexity, labels) in enumerate(self._hyps):
            # Score: primarily consistency, then simplicity (negative complexity).
            score = (self._consistency(labels), -complexity)
            if best_score is None or score > best_score:
                best_score = score
                candidates = [i]
            elif score == best_score:
                candidates.append(i)
        self._current = int(rng.choice(candidates))
