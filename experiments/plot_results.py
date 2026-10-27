"""Figures and statistics for the A2 report (Group 23).

Reads the per-seed CSVs written by A2_template_2026.py
(`<config>_seed<N>.csv`, columns: generation,best,mean) and produces the three
report figures plus every number quoted in the report:

  figure1_best_fitness.png/.pdf    : Figure 1, best fitness per generation
  figure2_final_boxplot.png/.pdf   : Figure 2, final best fitness of every run
  figure3_population_mean.png/.pdf : Figure 3, population mean fitness per generation

Usage:
    python plot_results.py                      # data/ next to code/, or __data__/A2_template_2026
    python plot_results.py --data path/to/csvs --out path/to/figures
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
DATA_CANDIDATES = [HERE.parent / "data", Path.cwd() / "__data__" / "A2_template_2026"]
POP_SIZE = 20  # evaluations per generation (must match A2_template_2026.py)
SEEDS = list(range(1, 11))

# config file prefix -> (label, matplotlib colour)
CONFIGS = {
    "sigma0.1": ("EA, σ = 0.1", "C0"),
    "sigma0.5": ("EA, σ = 0.5", "C1"),
    "random": ("Random search", "C2"),
}
EAS = ["sigma0.1", "sigma0.5"]


def load(data: Path, config: str, seeds: list[int]) -> np.ndarray:
    """Return an array of shape (n_seeds, n_generations, 3)."""
    runs = [
        np.loadtxt(data / f"{config}_seed{s}.csv", delimiter=",", skiprows=1)
        for s in seeds
    ]
    if len({len(r) for r in runs}) != 1:
        msg = f"{config}: runs have different lengths"
        raise ValueError(msg)
    return np.stack(runs)


# ---------------------------------------------------------------- figures -- #
def band(ax, gens, values, config) -> None:
    label, colour = CONFIGS[config]
    mean, std = values.mean(axis=0), values.std(axis=0, ddof=1)
    ax.plot(gens, mean, color=colour, label=label)
    ax.fill_between(gens, mean - std, mean + std, color=colour, alpha=0.2, linewidth=0)


def save(fig, out: Path, name: str) -> None:
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200)
    plt.close(fig)


def figure1(runs, out: Path, n: int) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for config, arr in runs.items():
        band(ax, arr[0, :, 0], arr[:, :, 1], config)
    ax.axhline(2.0, color="grey", linestyle="--", linewidth=1, label="Standing still")
    ax.set_title(f"Best fitness, mean ± std over {n} seeds")
    ax.set_xlabel(f"Generation (random search: blocks of {POP_SIZE} evaluations)")
    ax.set_ylabel("Best distance to target (m), lower is better")
    ax.legend(loc="lower left")
    save(fig, out, "figure1_best_fitness")


def figure2(runs, out: Path, seeds: list[int]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    rng = np.random.default_rng(0)  # horizontal jitter only
    finals = [arr[:, -1, 1] for arr in runs.values()]
    ax.boxplot(finals, widths=0.5, showfliers=False, showmeans=True,
               medianprops={"color": "black"},
               meanprops={"marker": "D", "markerfacecolor": "white",
                          "markeredgecolor": "black", "markersize": 6})
    for i, (config, f) in enumerate(zip(runs, finals), start=1):
        x = i + rng.uniform(-0.15, 0.15, size=len(f))
        ax.scatter(x, f, s=30, color=CONFIGS[config][1], edgecolor="white",
                   linewidth=0.7, zorder=3)
        if config == "sigma0.1":  # name the two weakest small-step runs
            for j in np.argsort(f)[-2:]:
                ax.annotate(f"seed {seeds[j]}", (x[j], f[j]), xytext=(8, 0),
                            textcoords="offset points", va="center", fontsize=9)
    ax.axhline(2.0, color="grey", linestyle="--", linewidth=1)
    ax.set_xticks(range(1, len(runs) + 1), [CONFIGS[c][0] for c in runs])
    ax.set_title(f"Final best fitness of each run ({len(seeds)} seeds)")
    ax.set_ylabel("Final distance to target (m), lower is better")
    save(fig, out, "figure2_final_boxplot")


def figure3(runs, out: Path, n: int) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for config in EAS:
        band(ax, runs[config][0, :, 0], runs[config][:, :, 2], config)
    ax.set_title(f"Population mean fitness, mean ± std over {n} seeds")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Population mean distance (m)")
    ax.legend(loc="upper right")
    save(fig, out, "figure3_population_mean")


# ------------------------------------------------------------- statistics -- #
def cliffs_delta(x, y) -> float:
    """Positive when x tends to be LOWER (better) than y."""
    return float(np.sign(np.asarray(y)[None, :] - np.asarray(x)[:, None]).mean())


def report(runs, seeds: list[int]) -> None:
    best = {c: arr[:, :, 1] for c, arr in runs.items()}
    last = next(iter(best.values())).shape[1] - 1
    print(f"budget: {POP_SIZE * (last + 1)} evaluations per run, seeds {seeds}\n")

    print("Table 2 - final best fitness (m)")
    for c, b in best.items():
        f = b[:, -1]
        print(f"  {c:9s} mean {f.mean():.3f}  sd {f.std(ddof=1):.3f}  median {np.median(f):.3f}"
              f"  best {f.min():.3f}  worst {f.max():.3f}")

    print("\nTable 3 - two-sided Mann-Whitney U (SciPy default) and Cliff's delta")
    for a, b in [("sigma0.1", "random"), ("sigma0.5", "random"), ("sigma0.1", "sigma0.5")]:
        x, y = best[a][:, -1], best[b][:, -1]
        p = stats.mannwhitneyu(x, y, alternative="two-sided").pvalue
        print(f"  {a} vs {b}: p {p:.4f}  delta {cliffs_delta(x, y):+.2f}"
              f"  ({a} lower on {int((x < y).sum())}/{len(x)} seeds)")

    print("\nTable 4 - per-seed final best (initial = best of the shared generation-0 population)")
    for i, s in enumerate(seeds):
        print(f"  seed {s:2d}: initial {best['sigma0.1'][i, 0]:.3f}  "
              + "  ".join(f"{c} {best[c][i, -1]:.3f}" for c in best))

    print("\nTable 5 - mean best fitness at checkpoints")
    for g in (0, 100, 250, 400, last):
        print(f"  gen {g:3d}: " + "  ".join(f"{c} {best[c][:, g].mean():.3f}" for c in best))

    print("\nIn-text numbers")
    for c in EAS:
        print(f"  {c}: improvement of mean best over the last 100 generations "
              f"{best[c][:, -101].mean() - best[c][:, -1].mean():.3f} m")
    for c in EAS:
        gap = (runs[c][:, -1, 2] - runs[c][:, -1, 1]).mean()
        print(f"  {c}: population mean minus population best at generation {last}: {gap:.3f} m")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()
    data = args.data or next((d for d in DATA_CANDIDATES if d.is_dir()), None)
    if data is None:
        parser.error("no data folder found; pass --data")
    out = args.out or HERE.parent / "figures"
    out.mkdir(parents=True, exist_ok=True)

    runs = {c: load(data, c, args.seeds) for c in CONFIGS}
    figure1(runs, out, len(args.seeds))
    figure2(runs, out, args.seeds)
    figure3(runs, out, len(args.seeds))
    report(runs, args.seeds)
    print(f"\nfigures saved to {out}")


if __name__ == "__main__":
    main()
