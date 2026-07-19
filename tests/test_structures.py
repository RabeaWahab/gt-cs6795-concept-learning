"""Self-verifying tests for the SHJ stimuli and the six category structures.

Rather than trust a transcribed table, these tests *derive* the six category
types from first principles (exhaustive enumeration of the cube's bipartitions
under its symmetry group) and check that the hand-entered structures in
``concept_learning.structures`` match the genuine SHJ signatures.
"""

from __future__ import annotations

from itertools import combinations, permutations, product

import numpy as np
import pytest

from concept_learning import (
    N_STIMULI,
    STIMULI,
    all_structures,
    feature_matrix,
    get_structure,
    stimulus_index,
)
from concept_learning.structures import TYPES

# --------------------------------------------------------------------------- #
# Structural invariants, computed directly on the cube of stimuli.
# --------------------------------------------------------------------------- #

_VERTICES = list(product((0, 1), repeat=3))


def _neighbors(s):
    """Hamming-distance-1 neighbours of a vertex."""
    return [tuple(s[j] ^ (1 if k == j else 0) for j in range(3)) for k in range(3)]


def _relevant_dims(a_set):
    """How many dimensions actually affect category membership."""
    count = 0
    for k in range(3):
        flips_matter = any(
            (tuple(s[j] ^ (1 if j == k else 0) for j in range(3)) in a_set) != (s in a_set)
            for s in _VERTICES
        )
        count += flips_matter
    return count


def _within_category_edges(a_set):
    """Cube edges whose endpoints share a category (cohesion measure)."""
    w = 0
    for s in _VERTICES:
        for n in _neighbors(s):
            if s < n and ((s in a_set) == (n in a_set)):
                w += 1
    return w


def _linearly_separable(a_set):
    """Is there a linear boundary (integer weights + threshold) splitting A|B?"""
    for w in product(range(-3, 4), repeat=3):
        for th_half in range(-9, 10):
            th = th_half * 0.5
            if all((sum(w[i] * s[i] for i in range(3)) > th) == (s in a_set)
                   for s in _VERTICES):
                return True
    return False


def _signature(a_set):
    return (_relevant_dims(a_set), _linearly_separable(a_set), _within_category_edges(a_set))


# The unique structural signature of each SHJ type (rel_dims, lin_sep, W-edges).
_EXPECTED_SIGNATURE = {
    "I":   (1, True,  8),
    "II":  (2, False, 4),
    "III": (3, False, 4),
    "IV":  (3, True,  6),
    "V":   (3, False, 6),
    "VI":  (3, False, 0),
}


def _category_a_set(structure):
    return {STIMULI[i] for i in structure.category_a}


# --------------------------------------------------------------------------- #
# Stimuli
# --------------------------------------------------------------------------- #

def test_eight_unique_stimuli():
    assert len(STIMULI) == N_STIMULI == 8
    assert len(set(STIMULI)) == 8


def test_stimulus_index_roundtrip():
    for i, s in enumerate(STIMULI):
        assert stimulus_index(s) == i


def test_feature_matrix_codings():
    m01 = feature_matrix("01")
    assert m01.shape == (8, 3)
    assert set(np.unique(m01)) == {0.0, 1.0}
    mpm = feature_matrix("pm1")
    assert set(np.unique(mpm)) == {-1.0, 1.0}
    # pm1 is a faithful affine recoding of 01
    assert np.array_equal(mpm, 2.0 * m01 - 1.0)


def test_feature_matrix_rejects_bad_coding():
    with pytest.raises(ValueError):
        feature_matrix("nope")


# --------------------------------------------------------------------------- #
# The six structures
# --------------------------------------------------------------------------- #

def test_there_are_exactly_six_types():
    assert TYPES == ("I", "II", "III", "IV", "V", "VI")
    assert len(all_structures()) == 6


def test_each_structure_splits_four_four():
    for st in all_structures():
        assert st.labels.shape == (8,)
        assert set(np.unique(st.labels)) <= {0, 1}
        assert len(st.category_a) == 4
        assert len(st.category_b) == 4


@pytest.mark.parametrize("name", TYPES)
def test_structure_matches_its_shj_signature(name):
    """Each encoded type has the structural invariants of the genuine SHJ type."""
    st = get_structure(name)
    assert _signature(_category_a_set(st)) == _EXPECTED_SIGNATURE[name]


def test_enumeration_yields_exactly_six_classes_matching_ours():
    """Exhaustively confirm there are 6 structure classes and we cover all of them.

    Two bipartitions are equivalent under relabelling dimensions (3! perms),
    negating any dimensions (2^3), and swapping the categories. Grouping all
    C(8,4)/2 = 35 bipartitions by that symmetry must yield exactly six classes,
    one per encoded type, with matching signatures.
    """
    idx = {s: i for i, s in enumerate(_VERTICES)}
    syms = []
    for perm in permutations(range(3)):
        for neg in product((0, 1), repeat=3):
            syms.append(lambda s, perm=perm, neg=neg:
                        tuple(s[perm[j]] ^ neg[j] for j in range(3)))

    def canonical(a_tuple):
        best = None
        b = set(_VERTICES) - set(a_tuple)
        for cat in (set(a_tuple), b):
            for g in syms:
                key = tuple(sorted(idx[g(s)] for s in cat))
                if best is None or key < best:
                    best = key
        return best

    orbits, seen = {}, set()
    for a in combinations(_VERTICES, 4):
        ka = frozenset(a)
        if ka in seen or frozenset(_VERTICES) - ka in seen:
            continue
        seen.add(ka)
        orbits.setdefault(canonical(a), []).append(a)

    assert len(orbits) == 6

    # Every one of our six types lands in a distinct orbit, and the six orbits
    # are exactly covered.
    our_canon = {name: canonical(tuple(_category_a_set(get_structure(name))))
                 for name in TYPES}
    assert len(set(our_canon.values())) == 6
    assert set(our_canon.values()) == set(orbits)


def test_human_difficulty_ordering_and_tie():
    ranks = {st.name: st.human_rank for st in all_structures()}
    # classic ordering I < II < {III = IV = V} < VI
    assert ranks["I"] < ranks["II"] < ranks["III"] < ranks["VI"]
    assert ranks["III"] == ranks["IV"] == ranks["V"]  # the intermediate tie
