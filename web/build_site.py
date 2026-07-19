"""Inject precomputed results into the site template -> self-contained index.html.

Keeps the presentation site's numbers identical to the notebook's: both read the
same results.json. Re-run after regenerating results:

    python web/build_site.py
"""

from __future__ import annotations

import pathlib

HERE = pathlib.Path(__file__).parent
template = (HERE / "index.template.html").read_text()
results = (HERE / "results.json").read_text().strip()

if "/*__RESULTS_JSON__*/" not in template:
    raise SystemExit("placeholder /*__RESULTS_JSON__*/ not found in template")

out = template.replace("/*__RESULTS_JSON__*/", results)
(HERE / "index.html").write_text(out)
print(f"wrote {HERE / 'index.html'}  ({len(out):,} bytes)")
