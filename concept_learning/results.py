"""Compute the full result set once and export it as plain data.

Both the analysis notebook and the presentation website read from this single
source of truth, so their numbers can never drift apart. Run as a script to
(re)generate the JSON the website embeds:

    python -m concept_learning.results --out web/results.json --subjects 800
"""

from __future__ import annotations

import argparse
import json

from .experiment import run_experiment
from .models import ALCOVEModel, PrototypeModel, RuleModel
from .structures import TYPES, all_structures, HUMAN_DIFFICULTY_RANK

# The four model conditions we compare. The two ALCOVE rows differ only by the
# attention learning rate -- that contrast is the Type II experiment.
CONDITIONS: dict[str, tuple[str, callable]] = {
    "exemplar_attn_on":  ("Exemplar (ALCOVE)",           lambda: ALCOVEModel()),
    "exemplar_attn_off": ("Exemplar (attention off)",    lambda: ALCOVEModel(lambda_a=0.0)),
    "prototype":         ("Prototype",                   lambda: PrototypeModel()),
    "rule_v1":           ("Rule (simple)",               lambda: RuleModel()),
}

# One-line theory gloss shown in the UI.
CONDITION_BLURB = {
    "exemplar_attn_on":  "Stores every instance; learns which dimensions to attend to.",
    "exemplar_attn_off": "Same exemplar memory, but attention is frozen uniform.",
    "prototype":         "Stores one average per category; classifies by similarity to it.",
    "rule_v1":           "Searches for the simplest logical rule that fits.",
}


def compute_all_results(n_subjects: int = 800, n_blocks: int = 25,
                        seed: int = 0) -> dict:
    structures = all_structures()
    models: dict[str, dict] = {}
    for key, (label, factory) in CONDITIONS.items():
        per_type: dict[str, dict] = {}
        for st in structures:
            r = run_experiment(factory, st, n_subjects=n_subjects,
                               n_blocks=n_blocks, seed=seed)
            per_type[st.name] = {
                "total_errors": round(r.mean_total_errors, 3),
                "errors_per_block": [round(x, 4) for x in r.mean_errors_per_block.tolist()],
                "p_reached_criterion": round(r.p_reached_criterion, 4),
            }
        models[key] = {"label": label, "blurb": CONDITION_BLURB[key],
                       "per_type": per_type}

    return {
        "meta": {
            "n_subjects": n_subjects,
            "n_blocks": n_blocks,
            "seed": seed,
            "note": "Errors are mean per simulated subject; lower = easier to learn.",
        },
        "types": list(TYPES),
        "human_rank": HUMAN_DIFFICULTY_RANK,
        "human_ordering": "I < II < {III = IV = V} < VI",
        "models": models,
    }


def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="web/results.json")
    ap.add_argument("--subjects", type=int, default=800)
    ap.add_argument("--blocks", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = compute_all_results(args.subjects, args.blocks, args.seed)

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)

    # Console summary: the difficulty ordering per model.
    for key, m in data["models"].items():
        order = sorted(data["types"], key=lambda t: m["per_type"][t]["total_errors"])
        print(f"{m['label']:<28} " + " < ".join(order))
    print(f"\nwrote {args.out}  ({args.subjects} subjects x {args.blocks} blocks)")


if __name__ == "__main__":
    _main()
