"""Behavioural tests for the three models.

These assert the *qualitative* signatures each theory is supposed to produce on
the SHJ structures -- the findings the project turns on -- using modest subject
counts and loose, seed-fixed thresholds so they stay fast and stable.
"""

from __future__ import annotations

import numpy as np
import pytest

from concept_learning.models import ALCOVEModel, PrototypeModel, RuleModel
from concept_learning.experiment import run_experiment, run_subject
from concept_learning.structures import get_structure, all_structures

N_SUBJECTS = 120
N_BLOCKS = 20


def _errors_by_type(make_model, seed=7):
    return {
        st.name: run_experiment(make_model, st, n_subjects=N_SUBJECTS,
                                n_blocks=N_BLOCKS, seed=seed).mean_total_errors
        for st in all_structures()
    }


def _crit_by_type(make_model, seed=7):
    return {
        st.name: run_experiment(make_model, st, n_subjects=N_SUBJECTS,
                                n_blocks=N_BLOCKS, seed=seed).p_reached_criterion
        for st in all_structures()
    }


# --- interface ------------------------------------------------------------- #

@pytest.mark.parametrize("model", [PrototypeModel(), ALCOVEModel(), RuleModel()])
def test_p_a_is_a_probability(model):
    rng = np.random.default_rng(0)
    model.reset(rng)
    for s in range(8):
        assert 0.0 <= model.p_a(s) <= 1.0


def test_runner_terminates_on_prototype_type_II():
    """The horizon-bounded runner must return even when criterion is never met."""
    res = run_subject(PrototypeModel(), get_structure("II"),
                      np.random.default_rng(0), n_blocks=15)
    assert res.errors_per_block.shape == (15,)
    assert not res.reached_criterion  # prototypes coincide -> chance forever


# --- prototype: solves linearly separable, fails XOR/parity ---------------- #

def test_prototype_fails_xor_and_parity():
    crit = _crit_by_type(lambda: PrototypeModel())
    assert crit["I"] > 0.9            # unidimensional: easy
    assert crit["IV"] > 0.7           # linearly separable family resemblance: ok
    assert crit["II"] < 0.3           # XOR: prototypes coincide -> cannot learn
    assert crit["VI"] < 0.3           # parity: likewise


# --- ALCOVE: reproduces the human ordering incl. the Type II advantage ------ #

def test_alcove_reproduces_human_ordering():
    e = _errors_by_type(lambda: ALCOVEModel())
    assert e["I"] == min(e.values())                       # Type I easiest
    assert e["VI"] == max(e.values())                      # Type VI hardest
    # Type II advantage: easier than the III/IV/V band average and than VI.
    band = np.mean([e["III"], e["IV"], e["V"]])
    assert e["II"] < band
    assert e["II"] < e["VI"]


def test_attention_is_what_buys_the_type_II_advantage():
    on = _errors_by_type(lambda: ALCOVEModel())               # lambda_a > 0
    off = _errors_by_type(lambda: ALCOVEModel(lambda_a=0.0))  # attention frozen
    band_off = np.mean([off["III"], off["IV"], off["V"]])
    # With attention on, II beats the band; with attention off it does not.
    assert on["II"] < np.mean([on["III"], on["IV"], on["V"]])
    assert off["II"] >= band_off
    # Turning attention off makes Type II harder.
    assert off["II"] > on["II"]


# --- rule v1: trivial on Type I, but finds Type II hard -------------------- #

def test_rule_solves_type_I_but_not_type_II():
    e = _errors_by_type(lambda: RuleModel())
    assert e["I"] < 10             # a unidimensional rule is found almost at once
    assert e["II"] > 30            # XOR has no rule in the v1 space -> hard
    assert e["II"] > 3 * e["I"]    # the diagnostic contrast with humans
