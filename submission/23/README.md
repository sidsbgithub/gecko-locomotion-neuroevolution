# Group 23 – Evolutionary Computing 2026, Standard Assignment Task II

Report: `23.pdf`

## Contents

| Path | What it is |
|---|---|
| `code/A2_template_2026.py` | The experiment: gecko in SimpleFlatWorld, 17→6→6 tanh controller, (μ+λ) EA with Gaussian mutation, and the random-search baseline. Built on `ariel.ec`; `src/ariel` is unmodified. |
| `code/plot_results.py` | Produces Figures 1–3 and every statistic in the report (Tables 2–5 and the in-text numbers) from the CSVs. |
| `data/` | The 30 runs used in the report: `<config>_seed<N>.csv` for `sigma0.1`, `sigma0.5` and `random`, seeds 1–10. Columns: `generation,best,mean` (best and population mean distance to the target, in metres; for random search `best` is the best-so-far and `mean` the mean of each fresh block of 20). |
| `figures/` | Output of `plot_results.py`: `figure1_best_fitness` (report Figure 1), `figure2_final_boxplot` (Figure 2) and `figure3_population_mean` (Figure 3). |

## Reproducing the experiment

`A2_template_2026.py` needs the course fork of ariel
(https://github.com/AndrzejSzczepura/EvolutionaryComputing2026, Python 3.12).
Put it in `assignments/assignment_2/` of that repository and run it from the
repository root; results are written to `__data__/A2_template_2026/` under the
working directory.

```bash
uv run assignments/assignment_2/A2_template_2026.py <seed> <sigma|random>
uv run assignments/assignment_2/A2_template_2026.py 3 0.1      # EA, σ = 0.1, seed 3
uv run assignments/assignment_2/A2_template_2026.py 3 random   # random search, seed 3
```

All ten seeds of one configuration:

```bash
for s in 1 2 3 4 5 6 7 8 9 10; do uv run assignments/assignment_2/A2_template_2026.py $s 0.5; done
```

Each run is deterministic given its seed: the seed sets both NumPy's generator
and ariel.ec's generator (`set_seed`).

## Reproducing the figures and statistics

Needs numpy, scipy and matplotlib. From this folder:

```bash
python code/plot_results.py
```

It reads `data/` and writes the figures to `figures/`.
