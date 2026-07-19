"""The six Shepard, Hovland & Jenkins (1961) category structures.

Each structure splits the eight :data:`~concept_learning.stimuli.STIMULI` into
two equal categories (four each). Up to relabelling of the dimensions and the
category names, there are exactly six such structures -- a fact we do not take
on faith but *verify* by exhaustive enumeration in ``tests/test_structures.py``.

Each of the six types is uniquely identified by a small set of structural
invariants (see the enumeration test):

======  ==========  =================  ==================  =========================
Type    rel. dims   linearly sep.?     within-cat. edges   informal description
======  ==========  =================  ==================  =========================
I       1           yes                8                   one dimension decides
II      2           no                 4                   XOR of two dims (3rd irrel.)
III     3           no                 4                   rule + exceptions
IV      3           yes                6                   family resemblance (majority)
V       3           no                 6                   rule + exceptions
VI      3           no                 0                   parity of all three dims
======  ==========  =================  ==================  =========================

Human learning difficulty (errors to criterion) follows the classic ordering

    I  <  II  <  III  =  IV  =  V  <  VI

replicated by Nosofsky et al. (1994). Types III, IV and V are statistically
*tied*; reproducing that tie (a single intermediate band), not their internal
order, is what the benchmark asks of a model.

Note on the III / V labels
--------------------------
Types III and V are the two 3-relevant, non-linearly-separable, non-parity
structures (they differ in cohesion: 4 vs 6 within-category edges). Which one
is called "III" vs "V" is a labelling convention that varies across sources;
because the human data place them in the same tied band, the choice cannot
affect any result reported here. The assignment below is documented and fixed
for reproducibility; :data:`HUMAN_DIFFICULTY_RANK` encodes the tie explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .stimuli import STIMULI, stimulus_index

# --- Category A membership, given as the {0,1} stimulus bit-strings ----------
# Category A is coded as label 0, category B as label 1. Everything else is
# derived from these six sets, so this is the single source of truth.
_CATEGORY_A: dict[str, tuple[str, ...]] = {
    "I":   ("000", "001", "010", "011"),  # A iff dim-1 == 0            (1 dim)
    "II":  ("000", "001", "110", "111"),  # A iff dim-1 == dim-2 (XOR)  (2 dims)
    "III": ("000", "001", "010", "111"),  # rule + exceptions, W = 4    (3 dims)
    "IV":  ("000", "001", "010", "100"),  # majority of zeros (lin.sep) (3 dims)
    "V":   ("000", "001", "010", "101"),  # rule + exceptions, W = 6    (3 dims)
    "VI":  ("000", "011", "101", "110"),  # even parity of all dims     (3 dims)
}

#: Canonical ordering of the six types.
TYPES: tuple[str, ...] = ("I", "II", "III", "IV", "V", "VI")

#: Human difficulty rank (1 = easiest). Types III/IV/V share rank 3 (tied),
#: so equal ranks encode the empirical tie rather than a spurious ordering.
HUMAN_DIFFICULTY_RANK: dict[str, int] = {
    "I": 1, "II": 2, "III": 3, "IV": 3, "V": 3, "VI": 4,
}


def _labels_from_category_a(members: tuple[str, ...]) -> np.ndarray:
    """Turn a set of category-A bit-strings into a length-8 label vector.

    Labels are aligned to :data:`~concept_learning.stimuli.STIMULI` order:
    entry ``i`` is 0 if stimulus ``i`` is in category A, else 1.
    """
    a_indices = {stimulus_index(tuple(int(c) for c in s)) for s in members}
    if len(a_indices) != 4:
        raise ValueError(f"category A must contain 4 distinct stimuli, got {members!r}")
    return np.array([0 if stimulus_index(s) in a_indices else 1 for s in STIMULI],
                    dtype=int)


#: type name -> length-8 label vector (0 = category A, 1 = category B),
#: aligned to STIMULI order.
LABELS: dict[str, np.ndarray] = {
    name: _labels_from_category_a(members) for name, members in _CATEGORY_A.items()
}


@dataclass(frozen=True)
class CategoryStructure:
    """One SHJ category-learning problem."""

    name: str                 # "I" ... "VI"
    labels: np.ndarray        # length-8 vector of {0, 1}, aligned to STIMULI
    human_rank: int           # 1 (easiest) .. 4 (hardest); III/IV/V share 3

    @property
    def category_a(self) -> np.ndarray:
        """Indices of the four stimuli in category A (label 0)."""
        return np.flatnonzero(self.labels == 0)

    @property
    def category_b(self) -> np.ndarray:
        """Indices of the four stimuli in category B (label 1)."""
        return np.flatnonzero(self.labels == 1)


def get_structure(name: str) -> CategoryStructure:
    """Return the :class:`CategoryStructure` for a type name ("I".."VI")."""
    if name not in LABELS:
        raise KeyError(f"unknown SHJ type {name!r}; choose from {TYPES}")
    return CategoryStructure(name=name,
                             labels=LABELS[name].copy(),
                             human_rank=HUMAN_DIFFICULTY_RANK[name])


def all_structures() -> list[CategoryStructure]:
    """Return all six structures in canonical order."""
    return [get_structure(name) for name in TYPES]
