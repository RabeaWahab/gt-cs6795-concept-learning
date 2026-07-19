"""Experiment runner: train a model on an SHJ structure and score its learning.

Paradigm (following Nosofsky et al. 1994)
-----------------------------------------
A *block* presents all eight stimuli once in random order. On each trial the
model emits P(A), a response is **sampled** from it (probability matching), we
record whether it was correct, then the model updates from feedback. A simulated
*subject* is trained for up to ``n_blocks`` (a fixed horizon), and we average
over many subjects (random trial order + random initialisation each).

Why a fixed horizon (not "loop until criterion")
------------------------------------------------
Some model x structure cells never reach criterion -- e.g. the prototype model
on Types II and VI, whose category prototypes coincide, leaving it at chance
forever. An open-ended "train to criterion" loop would never terminate there.
So training is horizon-bounded; we report both total errors within the horizon
and whether/when criterion was reached. Humans solved every structure within the
cap, so a model's *failure* to is a finding, not a hang.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models.base import CategoryModel
from .stimuli import N_STIMULI
from .structures import CategoryStructure


@dataclass(frozen=True)
class SubjectResult:
    """Outcome of training one simulated subject on one structure."""

    errors_per_block: np.ndarray   # length n_blocks
    criterion_block: int | None    # 1-indexed block criterion was reached, else None

    @property
    def total_errors(self) -> int:
        return int(self.errors_per_block.sum())

    @property
    def reached_criterion(self) -> bool:
        return self.criterion_block is not None


@dataclass(frozen=True)
class ExperimentResult:
    """Aggregate over many subjects for one model x structure."""

    model_name: str
    structure_name: str
    human_rank: int
    mean_errors_per_block: np.ndarray   # length n_blocks
    mean_total_errors: float
    p_reached_criterion: float
    mean_criterion_block: float | None  # over subjects that reached it, else None
    n_subjects: int
    n_blocks: int


def run_subject(
    model: CategoryModel,
    structure: CategoryStructure,
    rng: np.random.Generator,
    n_blocks: int = 25,
    criterion_blocks: int = 1,
) -> SubjectResult:
    """Train one subject; return per-block errors and criterion block.

    ``criterion_blocks`` is the number of *consecutive* error-free blocks that
    counts as having learned the structure (default 1).
    """
    labels = structure.labels
    model.reset(rng)
    errors = np.zeros(n_blocks, dtype=int)
    criterion_block: int | None = None
    consecutive_clean = 0

    for b in range(n_blocks):
        order = rng.permutation(N_STIMULI)
        for s in order:
            s = int(s)
            p = model.p_a(s)
            response = 0 if rng.random() < p else 1
            true = int(labels[s])
            if response != true:
                errors[b] += 1
            model.update(s, response, true, rng)

        if errors[b] == 0:
            consecutive_clean += 1
            if criterion_block is None and consecutive_clean >= criterion_blocks:
                # First block of the clean run that met criterion.
                criterion_block = b + 1 - (criterion_blocks - 1)
        else:
            consecutive_clean = 0

    return SubjectResult(errors_per_block=errors, criterion_block=criterion_block)


def run_experiment(
    make_model,
    structure: CategoryStructure,
    n_subjects: int = 500,
    n_blocks: int = 25,
    criterion_blocks: int = 1,
    seed: int = 0,
) -> ExperimentResult:
    """Average many subjects for one model x structure.

    ``make_model`` is a zero-argument factory returning a fresh model, so each
    subject starts from an independent instance (defensive; ``reset`` also
    re-initialises). Reproducible given ``seed``.
    """
    rng = np.random.default_rng(seed)
    per_block = np.zeros((n_subjects, n_blocks), dtype=int)
    reached = np.zeros(n_subjects, dtype=bool)
    crit_blocks: list[int] = []
    name = make_model().name

    for i in range(n_subjects):
        res = run_subject(make_model(), structure, rng,
                          n_blocks=n_blocks, criterion_blocks=criterion_blocks)
        per_block[i] = res.errors_per_block
        reached[i] = res.reached_criterion
        if res.criterion_block is not None:
            crit_blocks.append(res.criterion_block)

    return ExperimentResult(
        model_name=name,
        structure_name=structure.name,
        human_rank=structure.human_rank,
        mean_errors_per_block=per_block.mean(axis=0),
        mean_total_errors=float(per_block.sum(axis=1).mean()),
        p_reached_criterion=float(reached.mean()),
        mean_criterion_block=(float(np.mean(crit_blocks)) if crit_blocks else None),
        n_subjects=n_subjects,
        n_blocks=n_blocks,
    )
