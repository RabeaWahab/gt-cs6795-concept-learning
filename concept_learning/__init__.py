"""concept_learning: prototype, exemplar (GCM) and rule models on the SHJ task.

A computational comparison of three theories of concept representation --
prototype, exemplar (Generalized Context Model), and rule-based -- evaluated on
whether they reproduce the human difficulty ordering of the six Shepard,
Hovland & Jenkins (1961) category structures.

This package currently provides the task foundation:

* :mod:`concept_learning.stimuli`     -- the eight SHJ stimuli and their coding
* :mod:`concept_learning.structures`  -- the six category structures + human data

The three models and the experiment runner are added on top of this.
"""

from __future__ import annotations

from .stimuli import (
    N_DIMENSIONS,
    N_STIMULI,
    STIMULI,
    feature_matrix,
    stimulus_index,
)
from .structures import (
    HUMAN_DIFFICULTY_RANK,
    LABELS,
    TYPES,
    CategoryStructure,
    all_structures,
    get_structure,
)

__all__ = [
    "N_DIMENSIONS",
    "N_STIMULI",
    "STIMULI",
    "feature_matrix",
    "stimulus_index",
    "TYPES",
    "LABELS",
    "HUMAN_DIFFICULTY_RANK",
    "CategoryStructure",
    "get_structure",
    "all_structures",
]
