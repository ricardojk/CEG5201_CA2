"""[B-n] [C-n] [D-n] [E]  Report tables (CSV) and figures built from the raw benchmark runs.

Input : <results>/<algorithm>/raw/seq.json and mp_p<p>.json   (written by common/bench.py)
Output: <results>/<algorithm>/tables/*.csv, <results>/<algorithm>/figures/*,
        <results>/comparison/*
File names follow the CA2 statement, with n = the algorithm number of Table II.
"""
import json
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")                   # render straight to files; no GUI backend needed
import matplotlib.pyplot as plt         # noqa: E402  (must come after matplotlib.use)
import pandas as pd                     # noqa: E402

import config                           # noqa: E402

# Categorical colours, validated for colour-blind separation on a white background.
# Algorithm n always uses slot n; process count p uses slot p.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h"]
INK, INK_SOFT, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


# ---------------------------------------------------------------------------
# Loading and derived quantities
# ---------------------------------------------------------------------------
@dataclass
class Runs:
    """Raw results of one algorithm.

    seq and every mp[p] are DataFrames indexed by group g, with one column per pair i,
    holding the measured cumulative times T_i (seconds) of that group.
    """
    name: str
    n: int
    title: str
    sizes: list
    seq: pd.DataFrame
    mp: dict
    pool_startup: dict      # p -> Series of pool start-up seconds, indexed by group


@dataclass
class Metrics:
    """The quantities defined in [B-n], [C-n] and [D-n] of the statement."""
    T: pd.Series            # T_i of G0, sequential                       (B-n.1)
    seq_totals: pd.Series   # T(G_g), sequential                          (B-n.2)
    Tp: pd.DataFrame        # T_i^(p) of G0: pair x p                     (C-n.1)
    mp_totals: pd.DataFrame # T^(p)(G_g): group x p                       (C-n.2)
    speedup_k: pd.DataFrame # S_p(k) = T_k / T_k^(p): k x p, k = 1..8     (D-n.1)
    speedup_g: pd.DataFrame # cumulative seq / cumulative mp: group x p   (D-n.2)


def _read(path):
    """One raw JSON file -> (record, cumulative times, pool start-up times), indexed by group.

    Indexing by group keeps runs of different length aligned (an interrupted run
    simply has fewer rows).
    """
    record = json.loads(path.read_text(encoding="utf-8"))
    groups = [g["group"] for g in record["groups"]]
    frame = pd.DataFrame([g["cumulative_s"] for g in record["groups"]], index=groups)
    startup = pd.Series([g["pool_startup_s"] for g in record["groups"]], index=groups)
    return record, frame, startup


def load(results_dir, name):
    """Collect the sequential run and every multiprocessing run of algorithm `name`."""
    raw = results_dir / name / "raw"
    record, seq, _ = _read(raw / "seq.json")
    mp, startup = {}, {}
    for path in raw.glob("mp_p*.json"):
        mp_record, frame, pool_startup = _read(path)
        mp[mp_record["p"]] = frame
        startup[mp_record["p"]] = pool_startup
    n = next(k for k, v in config.ALGORITHMS.items() if v == name)
    return Runs(name, n, config.ALGORITHM_TITLES[n], record["sizes"], seq,
                dict(sorted(mp.items())), dict(sorted(startup.items())))


def per_pair(cumulative):
    """Derived per-pair times: t_0 = T_0 and t_i = T_i - T_(i-1)."""
    return cumulative.diff().fillna(cumulative)


def metrics(runs):
    """Derive every reported quantity from the measured cumulative times."""
    T = runs.seq.loc[0]
    seq_totals = runs.seq.iloc[:, -1]      # the group total is T after its last pair
    Tp = pd.DataFrame({p: frame.loc[0] for p, frame in runs.mp.items()})
    mp_totals = pd.DataFrame({p: frame.iloc[:, -1] for p, frame in runs.mp.items()})
    speedup_k = Tp.rdiv(T, axis=0)
    speedup_k.index = speedup_k.index + 1  # k = number of pairs completed = pair index + 1
    speedup_g = mp_totals.cumsum().rdiv(seq_totals.cumsum(), axis=0)
    return Metrics(T, seq_totals, Tp, mp_totals, speedup_k, speedup_g)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def _wide(leading, blocks):
    """Report table: the leading columns, then one column per p for every block.

    blocks maps a column-name template containing "{p}" to a DataFrame with p columns.
    """
    table = pd.DataFrame(leading)
    for template, frame in blocks.items():
        for p in frame.columns:
            table[template.format(p=p)] = frame[p].to_numpy()
    return table


def tables(runs, m):
    """All tables of one algorithm, keyed by their name in the statement."""
    n, sizes = runs.n, runs.sizes
    out = {
        f"B-{n}.1": pd.DataFrame({
            "Pair index": m.T.index,
            "Size": sizes,
            "Measured cumulative elapsed time T_i (s)": m.T.to_numpy(),
            "Derived time for this pair t_i (s)": per_pair(m.T).to_numpy(),
        }),
        f"B-{n}.2": pd.DataFrame({
            "Group index": m.seq_totals.index,
            "Measured total time T(G_g) (s)": m.seq_totals.to_numpy(),
            "Cumulative time through group (s)": m.seq_totals.cumsum().to_numpy(),
        }),
    }
    if runs.mp:
        startup = pd.DataFrame(runs.pool_startup)
        out[f"C-{n}.1"] = _wide({"Pair index": m.Tp.index, "Size": sizes},
                                {"T_i^({p}) (s)": m.Tp, "t_i^({p}) (s)": per_pair(m.Tp)})
        out[f"C-{n}.2"] = _wide({"Group index": m.mp_totals.index},
                                {"T^({p})(G_g) (s)": m.mp_totals,
                                 "Cumulative through group p={p} (s)": m.mp_totals.cumsum()})
        out[f"C-{n}_pool_startup"] = pd.DataFrame({
            "p": startup.columns,
            "Mean pool start-up per group (s)": startup.mean().to_numpy(),
            "Max pool start-up per group (s)": startup.max().to_numpy(),
        })
        out[f"D-{n}.1"] = _wide({"k (pairs completed)": m.speedup_k.index, "Size of pair k": sizes},
                                {"S_{p}(k)": m.speedup_k})
        out[f"D-{n}.2"] = _wide({"Through group": m.speedup_g.index},
                                {"S_{p} cumulative": m.speedup_g})
    return out


# ---------------------------------------------------------------------------
# Figure helpers  (thin lines, ringed markers, hairline grid, muted text)
# ---------------------------------------------------------------------------
def _axes(xlabel, ylabel, title):
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_xlabel(xlabel, fontsize=9, color=INK_SOFT)
    ax.set_ylabel(ylabel, fontsize=9, color=INK_SOFT)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS)
    ax.tick_params(which="both", colors=INK_SOFT, labelsize=8.5)
    return fig, ax


def _line(ax, x, y, color, marker, label=None):
    ax.plot(x, y, color=color, marker=marker, label=label, linewidth=1.6, markersize=6,
            markeredgecolor="white", markeredgewidth=1.2,
            solid_capstyle="round", solid_joinstyle="round")


def _ideal(ax, ps):
    """Reference line of linear speed-up, S = p."""
    ax.plot(ps, ps, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate("ideal S = p", (ps[-1], ps[-1]), textcoords="offset points", xytext=(-6, 2),
                ha="right", va="bottom", fontsize=8, color=MUTED)


def _break_even(ax):
    """Reference line S = 1: below it, multiprocessing is slower than sequential."""
    ax.axhline(1, color=AXIS, linewidth=1)


def _save(fig, out_dir, name, legend=False):
    if legend:
        fig.axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK_SOFT)
    path = out_dir / f"{name}.{config.FIGURE_FORMAT}"
    fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _k_ticks(ax, ks, sizes):
    """Label workload axis ticks with k and the size of the k-th pair."""
    ax.set_xticks(ks, [f"{k}\n({size})" for k, size in zip(ks, sizes)])


# ---------------------------------------------------------------------------
# Figures of one algorithm  (D-n.1 (i), D-n.1 (ii), D-n.2)
# ---------------------------------------------------------------------------
def algorithm_figures(runs, m, out_dir):
    color, marker = SERIES[runs.n - 1], MARKERS[runs.n - 1]
    ps = list(m.speedup_k.columns)
    k_last, g_last = m.speedup_k.index[-1], m.speedup_g.index[-1]
    paths = []

    # (i) speed-up vs process count for the complete G0 workload
    fig, ax = _axes("Processes p", f"Speed-up $S_p({k_last})$",
                    f"{runs.title}: speed-up vs processes (G0, all {k_last} pairs)")
    _ideal(ax, ps)
    _line(ax, ps, m.speedup_k.loc[k_last], color, marker)
    ax.set_xticks(ps)
    paths.append(_save(fig, out_dir, f"D-{runs.n}.1_speedup_vs_p"))

    # (ii) speed-up vs workload k, one curve per process count
    fig, ax = _axes("k = pairs completed (size of pair k)", "Speed-up $S_p(k)$",
                    f"{runs.title}: speed-up vs workload (G0)")
    _break_even(ax)
    for p in ps:
        _line(ax, m.speedup_k.index, m.speedup_k[p], SERIES[p - 1], MARKERS[p - 1], f"p = {p}")
    _k_ticks(ax, m.speedup_k.index, runs.sizes)
    paths.append(_save(fig, out_dir, f"D-{runs.n}.1_speedup_vs_k", legend=True))

    # D-n.2: all-group speed-up next to the G0-only result
    fig, ax = _axes("Processes p", "Speed-up",
                    f"{runs.title}: speed-up vs processes, all groups")
    _ideal(ax, ps)
    _line(ax, ps, m.speedup_k.loc[k_last], MUTED, "o", "G0 only")
    _line(ax, ps, m.speedup_g.loc[g_last], color, marker, f"G0-G{g_last} cumulative")
    ax.set_xticks(ps)
    paths.append(_save(fig, out_dir, f"D-{runs.n}.2_speedup_vs_p", legend=True))
    return paths


def write_algorithm(runs, out_dir):
    """Write every table and figure of one algorithm; return the written paths."""
    m = metrics(runs)
    table_dir = out_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, table in tables(runs, m).items():
        path = table_dir / f"{name}.csv"
        table.to_csv(path, index=False, float_format="%.6g")
        paths.append(path)
    if runs.mp:
        figure_dir = out_dir / "figures"
        figure_dir.mkdir(exist_ok=True)
        paths += algorithm_figures(runs, m, figure_dir)
    return paths


# ---------------------------------------------------------------------------
# [E] Comparison of the algorithms
# ---------------------------------------------------------------------------
def write_comparison(all_runs, out_dir):
    """Cross-algorithm figures and summary table; every run needs seq and mp results."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ms = {runs.name: metrics(runs) for runs in all_runs}
    sizes = all_runs[0].sizes
    p_common = min(m.speedup_k.columns[-1] for m in ms.values())
    all_ps = sorted(set().union(*(m.speedup_g.columns for m in ms.values())))
    paths = []

    fig, ax = _axes("Matrix size M", "Sequential time per pair (s)",
                    "Sequential time per multiplication (G0)")
    for runs in all_runs:
        _line(ax, runs.sizes, per_pair(ms[runs.name].T), SERIES[runs.n - 1],
              MARKERS[runs.n - 1], runs.title)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(sizes, [str(size) for size in sizes])
    ax.set_xticks([], minor=True)
    paths.append(_save(fig, out_dir, "E_seq_time_vs_size", legend=True))

    fig, ax = _axes("Processes p", "Speed-up (all groups, cumulative)",
                    "Speed-up vs processes, all groups")
    _ideal(ax, all_ps)
    for runs in all_runs:
        speedup = ms[runs.name].speedup_g.iloc[-1]
        _line(ax, speedup.index, speedup, SERIES[runs.n - 1], MARKERS[runs.n - 1], runs.title)
    ax.set_xticks(all_ps)
    paths.append(_save(fig, out_dir, "E_speedup_vs_p", legend=True))

    fig, ax = _axes("k = pairs completed (size of pair k)", f"Speed-up $S_{{{p_common}}}(k)$",
                    f"Speed-up vs workload at p = {p_common} (G0)")
    _break_even(ax)
    for runs in all_runs:
        speedup = ms[runs.name].speedup_k[p_common]
        _line(ax, speedup.index, speedup, SERIES[runs.n - 1], MARKERS[runs.n - 1], runs.title)
    _k_ticks(ax, ms[all_runs[0].name].speedup_k.index, sizes)
    paths.append(_save(fig, out_dir, "E_speedup_vs_k", legend=True))

    rows = []
    for runs in all_runs:
        m = ms[runs.name]
        final_k, final_g = m.speedup_k.iloc[-1], m.speedup_g.iloc[-1]
        p_max = final_g.index[-1]
        rows.append({
            "Algorithm": runs.title,
            "n": runs.n,
            "Sequential time G0 (s)": m.T.iloc[-1],
            "Sequential time all groups (s)": m.seq_totals.sum(),
            "Best p on G0": final_k.idxmax(),
            "Best speed-up on G0": final_k.max(),
            "Max p": p_max,
            "Speed-up at max p, all groups": final_g[p_max],
            "Efficiency at max p, all groups": final_g[p_max] / p_max,
        })
    path = out_dir / "E_summary.csv"
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.6g")
    paths.append(path)
    return paths
