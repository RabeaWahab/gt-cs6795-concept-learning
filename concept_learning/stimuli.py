"""Shepard, Hovland & Jenkins (1961) stimuli.

The SHJ paradigm uses eight stimuli formed by fully crossing three binary
dimensions (2^3 = 8). Concretely, Nosofsky et al. (1994) used shape
(square/triangle), interior line (solid/dotted), and size (large/small).
The *abstract* structure is all that matters for the models, so we work with
the binary feature vectors and stay agnostic about the physical realisation.

Feature coding
--------------
Each stimulus is a length-3 vector over {0, 1}. We keep the canonical binary
ordering 000, 001, 010, ..., 111 so a stimulus's index equals the integer its
bit-string encodes (000 -> 0, 111 -> 7). Models that prefer a symmetric
psychological space (e.g. the GCM's distance metric) can request the {-1, +1}
encoding via :func:`feature_matrix`.

References
----------
Shepard, R. N., Hovland, C. I., & Jenkins, H. M. (1961). Learning and
memorization of classifications. *Psychological Monographs*, 75(13), 1-42.

Nosofsky, R. M., Gluck, M. A., Palmeri, T. J., McKinley, S. C., & Glauthier, P.
(1994). Comparing models of rule-based classification learning: A replication
and extension of Shepard, Hovland, and Jenkins (1961). *Memory & Cognition*,
22(3), 352-369.
"""

from __future__ import annotations

from itertools import product

import numpy as np

#: Number of binary dimensions defining each stimulus.
N_DIMENSIONS: int = 3

#: Number of stimuli (2 ** N_DIMENSIONS).
N_STIMULI: int = 2 ** N_DIMENSIONS

#: The eight stimuli as {0, 1} tuples, in canonical binary order.
#: STIMULI[i] is the bit-string of the integer i, most-significant bit first.
STIMULI: tuple[tuple[int, int, int], ...] = tuple(
    product((0, 1), repeat=N_DIMENSIONS)
)


def stimulus_index(stimulus: tuple[int, ...]) -> int:
    """Return the canonical index (0-7) of a {0, 1} stimulus vector."""
    idx = 0
    for bit in stimulus:
        idx = (idx << 1) | int(bit)
    return idx


def feature_matrix(coding: str = "01") -> np.ndarray:
    """Return the 8x3 stimulus matrix.

    Parameters
    ----------
    coding:
        ``"01"`` returns features in {0, 1} (default).
        ``"pm1"`` returns them in {-1, +1}, a symmetric space that is often
        more convenient for similarity/distance-based models such as the GCM.
    """
    mat = np.array(STIMULI, dtype=float)
    if coding == "01":
        return mat
    if coding == "pm1":
        return 2.0 * mat - 1.0
    raise ValueError(f"unknown coding {coding!r}; use '01' or 'pm1'")
