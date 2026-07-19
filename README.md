# A Computational Comparison of Prototype, Exemplar, and Rule-Based Models of Human Category Learning

CS 6795: Cognitive Science (Georgia Tech, Summer 2026) — Rabea Abdelwahab

This project implements three competing theories of how a concept is represented
in the mind — a **prototype** model, an **exemplar** model (Nosofsky's Generalized
Context Model), and a **rule-based** model — and asks not which classifies most
accurately, but **which one generalizes the way people do**: reproducing the human
difficulty ordering across the six Shepard, Hovland & Jenkins (SHJ, 1961) category
structures.

Success is measured against the human learning-difficulty ordering

```
I  <  II  <  III  =  IV  =  V  <  VI
```

using **errors-to-criterion (learning curves)**, following the paradigm of
Nosofsky, Gluck, Palmeri, McKinley & Glauthier (1994). Reproducing this ordering
(especially the "unexpectedly easy" Type II) is treated as the cognitive claim;
raw accuracy is reported but is not the thing to win on.

## Status

| Component | Status |
|-----------|--------|
| SHJ stimuli + six category structures (`concept_learning/`) | ✅ implemented & tested |
| Three learning models (prototype, GCM/ALCOVE, rule) | ✅ implemented & tested |
| Experiment runner + learning curves (`experiment.py`) | ✅ implemented & tested |
| Precomputed results export (`results.py`) | ✅ |
| Interactive presentation website (`web/`) | ✅ built & published |
| Comparison figures & analysis notebook (`notebooks/`) | ✅ built & runs end-to-end |
| Final report — IEEE PDF (`report/`) | ✅ drafted & compiled (6 pp, 13 refs) |
| Rule model v2 (RULEX-style rule + exceptions) | ⬜ |
| Medin & Schaffer (1978) 5-4 secondary check | ⬜ |

### Result so far (300 simulated subjects/cell)

Errors-to-criterion ordering, easy → hard:

| Model | Ordering | Reproduces human `I < II < {III,IV,V} < VI`? |
|-------|----------|----------------------------------------------|
| **Exemplar — ALCOVE (attention on)** | I < II < V ≈ IV ≈ III < VI | ✅ yes, incl. the Type II advantage |
| Exemplar — ALCOVE (attention **off**) | I < V ≈ IV ≈ III < II < VI | Type II advantage collapses — attention is the mechanism |
| **Prototype** | I < IV < V < III < II ≈ VI | fails XOR/parity (cannot solve II, VI) |
| **Rule v1 (simple)** | I (≈1 error) < IV ≈ V ≈ III ≈ II < VI | trivial on I, but finds II *hard* — opposite of humans |

The headline: only the **exemplar model with learned selective attention** reproduces the human ordering, and the attention-off ablation shows attention is precisely what makes the "unexpectedly easy" Type II easy.

## Layout

```
concept_learning/      # the package
  stimuli.py           # the 8 SHJ stimuli (3 binary dims) and feature codings
  structures.py        # the 6 category structures + human difficulty ranks
tests/
  test_structures.py   # self-verifying: re-derives the 6 types by enumeration
notebooks/
  model_comparison.ipynb  # the core deliverable: runs all models, makes the figures
figures/                  # PNGs the notebook writes (fig0..fig3), reusable in the report
```

The six structures are not transcribed from a table; `test_structures.py`
exhaustively enumerates every 4|4 split of the stimulus cube, groups them into
the six symmetry classes, and checks each encoded type against the genuine SHJ
structural signature (relevant dimensions, linear separability, cohesion).

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pytest -q                                   # run the test suite
python -m concept_learning.results          # regenerate web/results.json
python web/build_site.py                    # rebuild the self-contained web/index.html
jupyter nbconvert --to notebook --execute --inplace \
    notebooks/model_comparison.ipynb        # run the core notebook + write figures/
```

Or just open `notebooks/model_comparison.ipynb` in Jupyter and Run All — it locates the
package itself, so it works from any working directory.

## Presentation site

`web/index.html` is a self-contained, interactive visualization (for the video
and poster) driven entirely by `web/results.json` — the *same* precomputed run
the notebook uses, so the numbers can't drift. It has a model selector, a live
selective-attention on/off toggle (watch Type II move), animated learning
curves, and the six-structure strip. Rebuild with the two commands above.

## References

- Shepard, R. N., Hovland, C. I., & Jenkins, H. M. (1961). Learning and
  memorization of classifications. *Psychological Monographs*, 75(13), 1-42.
- Nosofsky, R. M., Gluck, M. A., Palmeri, T. J., McKinley, S. C., & Glauthier, P.
  (1994). Comparing models of rule-based classification learning. *Memory &
  Cognition*, 22(3), 352-369.
- Medin, D. L., & Schaffer, M. M. (1978). Context theory of classification
  learning. *Psychological Review*, 85(3), 207-238.
- Nosofsky, R. M. (1986). Attention, similarity, and the identification-
  categorization relationship. *J. Exp. Psychol.: General*, 115(1), 39-61.
