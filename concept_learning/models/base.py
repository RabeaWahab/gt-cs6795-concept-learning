"""Common interface for the category-learning models.

All three models are **online learners**: on each trial they see a stimulus,
emit a probability of responding "category A", a response is *sampled* from that
probability, and then the model updates from corrective feedback. This mirrors
the trial-by-trial supervised learning of the Shepard, Hovland & Jenkins task
and lets every model be scored the same way (errors per block, errors to
criterion) by :mod:`concept_learning.experiment`.

Categories are coded as in :mod:`concept_learning.structures`: **0 = A, 1 = B**.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class CategoryModel(ABC):
    """Abstract online two-category learner over the SHJ stimuli."""

    #: Human-readable model name for figures/tables.
    name: str = "model"

    @abstractmethod
    def reset(self, rng: np.random.Generator) -> None:
        """Return the model to its naive, pre-training state.

        Called once per simulated subject. ``rng`` seeds any random
        initialisation so runs are reproducible.
        """

    @abstractmethod
    def p_a(self, stim_idx: int) -> float:
        """Probability in ``[0, 1]`` of responding category A for a stimulus."""

    @abstractmethod
    def update(self, stim_idx: int, response: int, true_label: int,
               rng: np.random.Generator) -> None:
        """Learn from one trial.

        Parameters
        ----------
        stim_idx:
            Index (0-7) of the presented stimulus.
        response:
            The category the model actually responded (0 or 1), sampled from
            :meth:`p_a`. Supervised models ignore this; the rule model's
            lose-shift needs it.
        true_label:
            The correct category (0 or 1) revealed as feedback.
        rng:
            Shared random generator for any stochastic update.
        """
