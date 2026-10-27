# Assignment 1 — evolving robot bodies towards a target set (tree genome)

Code for the report. Everything runs from the repository root inside the
project's virtual environment (Python 3.12, Ariel fork from the course
repository with `src/ariel` unmodified, NumPy, pandas, SciPy, matplotlib).

## Files

| File | Purpose |
| --- | --- |
| `A1_evolution_tree.py` | The EA and all experiment modes (final experiment, screening, random search) |
| `A1_final_analysis.py` | Figures and statistics for the final experiment |
| `tree_edit_distance.py` | The assignment's fitness function (unchanged starter code) |
| `target_bodies/` | The five target bodies (7, 11, 15, 19, 25 modules) |
| `A1_template_2026.py` | Course starter template (installation check) |
| `__data__/A1_evolution_tree/` | Generated results (CSV, PNG); not part of the code |

## Reproduce the reported experiment

```bash
python assignments/assignment_1/A1_evolution_tree.py --final --workers 10
```

Runs five EA variants that differ only in the mutation operator
(`subtree`, `mixed`, `replace`, `shrink`, `hoist`) with all other settings
fixed (mutation probability 0.75, one application, subtree crossover with
probability 1.0, truncation parent selection, (mu + lambda) survival,
population 60, 100 generations = 6,060 evaluations) on ten held-out seeds
(7001–7010), then a random-search baseline of 6,060 random bodies on the
same seeds, then the analysis. Output:

```
__data__/A1_evolution_tree/final_experiment/
    raw_results.csv          one row per run (final best fitness, best body size, ...)
    generation_results.csv   per-generation statistics for every run
    final_stats.csv          mean / std / median / 95 % CI per variant and for random search
    pairwise_tests.csv       Mann-Whitney U (two-sided) and Cliff's delta for every pair
    fig_convergence.png      best fitness per generation, mean +/- std, all groups
    fig_tree_size.png        population mean modules per generation vs the target sizes
    fig_final_boxplot.png    final best fitness per run
__data__/A1_evolution_tree/random_search/random_search_budget6060.csv
```

Finished runs found on disk are reused; add `--force-rerun` to recompute.
Every run is deterministic given its seed.

## Reproduce the parameter screening (how the fixed settings were chosen)

```bash
# Stage 1: one-factor-at-a-time screens of mutation, crossover and selection
python assignments/assignment_1/A1_evolution_tree.py --screen --screen-axis all --screen-pop 25 --screen-generations 20 --screen-seeds 42 123 456 --workers 4

# Stage 3: full mutation x crossover factorial (420 configurations x 8 seeds) with a
# second stage that re-runs the top 20 on 56 extra seeds and ranks them statistically
python assignments/assignment_1/A1_evolution_tree.py --screen --screen-axis factorial --screen-name factorial_final --screen-pop 60 --screen-generations 100 --screen-seeds 42 123 456 789 1011 2024 31415 27182 --screen-mutations replace subtree shrink hoist mixed --screen-rates 0.1 0.25 0.4 0.5 0.6 0.75 1.0 --screen-events 1 2 --screen-crossover-options none subtree --screen-crossover-rates 0.2 0.4 0.6 0.8 1.0 --screen-selection truncation --refine-top-k 20 --refine-seeds 5001 5002 5003 5004 5005 5006 5007 5008 5009 5010 5011 5012 5013 5014 5015 5016 5017 5018 5019 5020 5021 5022 5023 5024 5025 5026 5027 5028 5029 5030 5031 5032 5033 5034 5035 5036 5037 5038 5039 5040 5041 5042 5043 5044 5045 5046 5047 5048 5049 5050 5051 5052 5053 5054 5055 5056 --workers 11
```

The factorial takes a few hours on 11 worker processes; its outputs
(`summary.csv`, `final_ranking.csv`, `marginal_effects.csv`,
`interaction_*.csv`, `factorial_heatmap.png`, ...) land in
`__data__/A1_evolution_tree/factorial_final/`.

Options that exist only for these screening stages and are not used in the
final experiment: k-tournament (`--selection tournament --tournament-k`) and
linear-ranking (`--selection rank --rank-pressure`) parent selection, more
than one mutation application per offspring (`--mutation-events`), and a
crossover probability below 1 (`--crossover-rate`).

## Single run

```bash
python assignments/assignment_1/A1_evolution_tree.py --seed 42            # final configuration, subtree mutation
python assignments/assignment_1/A1_evolution_tree.py --seed 42 --mutation hoist
```
