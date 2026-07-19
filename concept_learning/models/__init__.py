"""The three category-learning models, each a theory of concept representation.

* :class:`~concept_learning.models.prototype.PrototypeModel` -- prototype theory
* :class:`~concept_learning.models.exemplar.ALCOVEModel`     -- exemplar theory (GCM/ALCOVE)
* :class:`~concept_learning.models.rule.RuleModel`           -- classical rule theory
"""

from __future__ import annotations

from .base import CategoryModel
from .exemplar import ALCOVEModel
from .prototype import PrototypeModel
from .rule import RuleModel

__all__ = ["CategoryModel", "PrototypeModel", "ALCOVEModel", "RuleModel"]
