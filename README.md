# Code and results for "Can regenerated antlers preserve information about previous contests? Anatomical constraints and an information model"

Authors: Abdullah Ansar (The University of Chicago) and Sajidan Rafsan (St. Olaf College).

Python 3.12 with numpy 2.5.3, scipy 1.18.1 and matplotlib 3.11.2. Random seed 7 (Monte Carlo checks use seeds 7 to 11).

Files
- `sim2.py`: computes the information model by numerical quadrature and writes `grid2.json`; also runs the Monte Carlo cross-check and the robustness cells. Run `python sim2.py` (about 20 s).
- `power.py`: sample sizes per male and per male-season; writes `power.json`.
- `make_figures2.py`: redraws Figures 1 to 5 from `grid2.json` (reads it from the same folder).
- `grid2.json`, `power.json`: the results reported in the paper.
- `figures/`: Figures 1 to 5 as PDF and PNG.

No empirical data were generated; all numbers are model output. Code and figures were produced with the assistance of AI (Claude Sonnet 5.5, Anthropic), as declared in the paper.

Licence: MIT for the code, CC BY 4.0 for figures and results.
