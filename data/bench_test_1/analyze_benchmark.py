import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

METALS = ["Zn", "Mg", "Mn", "Ca"]
COLORS = {"overall": "#111111", "Zn": "#d6249f", "Mg": "#e8710a", "Mn": "#7b4fb5", "Ca": "#2e9e4f"}
RUN_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#17becf"]
THRESHOLDS = (1.0, 2.0, 3.0)
ROT_BINS = [-1, 2, 5, 8, 1000]
ROT_LABELS = ["0-2", "3-5", "6-8", "9+"]


def parse_runs(specs):
    runs = []
    for spec in specs:
        if "=" in spec:
            label, path = spec.split("=", 1)
        else:
            path = spec
            label = os.path.basename(os.path.dirname(os.path.abspath(spec))) or "run"
        runs.append((label, pd.read_csv(path)))
    return runs


def wilson(k, n, z=1.96):
    if n == 0:
        return math.nan, math.nan
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return 100 * (centre - half), 100 * (centre + half)


def subsets(df):
    yield "overall", df
    for metal in METALS:
        part = df[df.metal == metal]
        if len(part):
            yield metal, part


def success_table(label, df):
    rows = []
    for group, part in subsets(df):
        scored = part[part.status == "ok"]
        row = {"run": label, "group": group, "n_complexes": len(part), "n_scored": len(scored)}
        for column, tag in (("top1_rmsd", "top1"), ("best_pose_rmsd", "bestpose")):
            values = part[column] if column in part else pd.Series(dtype=float)
            row["median_rmsd_" + tag] = float(scored[column].median()) if len(scored) else math.nan
            row["mean_rmsd_" + tag] = float(scored[column].mean()) if len(scored) else math.nan
            for t in THRESHOLDS:
                k = int((values <= t).sum())
                lo, hi = wilson(k, len(part))
                row["%s_le_%gA_pct" % (tag, t)] = 100 * k / len(part) if len(part) else math.nan
                row["%s_le_%gA_ci_lo" % (tag, t)] = lo
                row["%s_le_%gA_ci_hi" % (tag, t)] = hi
        rows.append(row)
    return rows


def cumulative(values, total, grid):
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    return np.array([(arr <= x).sum() / total for x in grid])


def plot_pose_accuracy(runs, out):
    grid = np.linspace(0, 10, 401)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, (column, title) in zip(axes, (("top1_rmsd", "Best score (top-ranked pose)"), ("best_pose_rmsd", "Best pose (lowest RMSD of any pose)"))):
        for i, (label, df) in enumerate(runs):
            ax.plot(grid, cumulative(df[column], len(df), grid), color=RUN_COLORS[i % len(RUN_COLORS)], lw=2, label=label)
        ax.axvline(2, color="grey", ls="--", lw=1)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("RMSD (A)")
        ax.set_title(title)
    axes[0].set_ylabel("Cumulative proportion")
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "pose_accuracy.png"), dpi=200)
    plt.close(fig)


def plot_by_metal(label, df, out):
    grid = np.linspace(0, 10, 401)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, (column, title) in zip(axes, (("top1_rmsd", "Best score"), ("best_pose_rmsd", "Best pose"))):
        for group, part in subsets(df):
            ax.plot(grid, cumulative(part[column], len(part), grid), color=COLORS[group], lw=2,
                    label="%s (n=%d)" % (group, len(part)))
        ax.axvline(2, color="grey", ls="--", lw=1)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("RMSD (A)")
        ax.set_title("%s - %s" % (label, title))
    axes[0].set_ylabel("Cumulative proportion")
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "pose_accuracy_by_metal_%s.png" % label), dpi=200)
    plt.close(fig)


def plot_success_by_metal(table, out):
    groups = ["overall"] + METALS
    labels = list(dict.fromkeys(table.run))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, (tag, title) in zip(axes, (("top1", "Best score"), ("bestpose", "Best pose"))):
        width = 0.8 / max(1, len(labels))
        for i, label in enumerate(labels):
            sub = table[table.run == label].set_index("group")
            vals = [sub[tag + "_le_2A_pct"].get(g, math.nan) for g in groups]
            ax.bar(np.arange(len(groups)) + i * width, vals, width, color=RUN_COLORS[i % len(RUN_COLORS)], label=label)
        ax.set_xticks(np.arange(len(groups)) + width * (len(labels) - 1) / 2)
        ax.set_xticklabels(groups)
        ax.set_ylim(0, 100)
        ax.set_title("%s: success at 2 A" % title)
    axes[0].set_ylabel("Success rate (%)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "success_by_metal.png"), dpi=200)
    plt.close(fig)


def plot_topn(runs, out):
    ns = [1, 3, 5, 10]
    columns = ["top1_rmsd", "rmsd_top3_min", "rmsd_top5_min", "rmsd_top10_min"]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    rows = []
    for i, (label, df) in enumerate(runs):
        vals = [100 * (df[c] <= 2).sum() / len(df) if c in df else math.nan for c in columns]
        ax.plot(ns, vals, "o-", color=RUN_COLORS[i % len(RUN_COLORS)], label=label)
        rows.append(dict(run=label, **{"top%d_pct" % n: v for n, v in zip(ns, vals)}))
    ax.set_xlabel("Number of top-ranked poses considered")
    ax.set_ylabel("Success at 2 A (%)")
    ax.set_ylim(0, 100)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "topn_success.png"), dpi=200)
    plt.close(fig)
    return pd.DataFrame(rows)


def scoring_power(label, df, out):
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (column, title) in zip(axes, (("top1_e_final", "E_final (kcal/mol)"), ("top1_e_per_heavy", "E_final per heavy atom (kcal/mol)"))):
        usable = df[(df.status == "ok") & df[column].notna() & df.logKd.notna()] if column in df else df.iloc[0:0]
        for group, part in subsets(usable):
            if len(part) >= 3:
                r, pr = pearsonr(part[column], part.logKd)
                rho, ps = spearmanr(part[column], part.logKd)
                rows.append({"run": label, "score": column, "group": group, "n": len(part),
                             "pearson_r": r, "pearson_p": pr, "spearman_rho": rho, "spearman_p": ps})
            if group != "overall":
                ax.scatter(part.logKd, part[column], s=14, color=COLORS[group], label=group, alpha=0.8)
                if len(part) >= 3:
                    k, b = np.polyfit(part.logKd, part[column], 1)
                    xs = np.array([part.logKd.min(), part.logKd.max()])
                    ax.plot(xs, k * xs + b, color=COLORS[group], lw=1.2)
        if len(usable) >= 3:
            k, b = np.polyfit(usable.logKd, usable[column], 1)
            xs = np.array([usable.logKd.min(), usable.logKd.max()])
            ax.plot(xs, k * xs + b, color=COLORS["overall"], lw=1.8, label="overall")
        ax.set_xlabel("logKd (experimental)")
        ax.set_ylabel(title)
        ax.set_title("%s - scoring power" % label)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "scoring_power_%s.png" % label), dpi=200)
    plt.close(fig)
    return rows


def rotor_table(runs, out):
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, (column, title) in zip(axes, (("top1_rmsd", "Best score"), ("best_pose_rmsd", "Best pose"))):
        for i, (label, df) in enumerate(runs):
            if "n_rot" not in df:
                continue
            bins = pd.cut(df.n_rot, ROT_BINS, labels=ROT_LABELS)
            vals = []
            for name in ROT_LABELS:
                part = df[bins == name]
                pct = 100 * (part[column] <= 2).sum() / len(part) if len(part) else math.nan
                vals.append(pct)
                rows.append({"run": label, "metric": column, "rotatable_bonds": name, "n": len(part), "success_pct": pct})
            width = 0.8 / len(runs)
            ax.bar(np.arange(len(ROT_LABELS)) + i * width, vals, width, color=RUN_COLORS[i % len(RUN_COLORS)], label=label)
        ax.set_xticks(np.arange(len(ROT_LABELS)) + 0.4 - 0.4 / len(runs))
        ax.set_xticklabels(ROT_LABELS)
        ax.set_xlabel("Rotatable bonds")
        ax.set_ylim(0, 100)
        ax.set_title("%s: success at 2 A" % title)
    axes[0].set_ylabel("Success rate (%)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "success_by_rotatable_bonds.png"), dpi=200)
    plt.close(fig)
    return pd.DataFrame(rows)


def plot_baselines(path, table, out):
    base = pd.read_csv(path).dropna(subset=["best_score_success", "best_pose_success"], how="all")
    ours = table[table.group == "overall"]
    names = list(base.method) + list(ours.run)
    score = list(base.best_score_success) + list(ours.top1_le_2A_pct)
    pose = list(base.best_pose_success) + list(ours.bestpose_le_2A_pct)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(names)), 4.5))
    ax.bar(x - 0.2, score, 0.4, label="Best score", color="#1f77b4")
    ax.bar(x + 0.2, pose, 0.4, label="Best pose", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Success at 2 A (%)")
    ax.set_ylim(0, 100)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "comparison_with_published.png"), dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate qmdock redocking results the way Cinaroglu and Timucin did")
    parser.add_argument("results", nargs="+")
    parser.add_argument("--out", default="analysis")
    parser.add_argument("--baselines")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    runs = parse_runs(args.results)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)

    table = pd.DataFrame([r for label, df in runs for r in success_table(label, df)])
    table.to_csv(os.path.join(args.out, "success_rates.csv"), index=False)

    plot_pose_accuracy(runs, args.out)
    plot_success_by_metal(table, args.out)
    topn = plot_topn(runs, args.out)
    topn.to_csv(os.path.join(args.out, "topn_success.csv"), index=False)
    rotors = rotor_table(runs, args.out)
    rotors.to_csv(os.path.join(args.out, "success_by_rotatable_bonds.csv"), index=False)

    power = []
    for label, df in runs:
        plot_by_metal(label, df, args.out)
        power.extend(scoring_power(label, df, args.out))
    power = pd.DataFrame(power)
    power.to_csv(os.path.join(args.out, "scoring_power.csv"), index=False)

    if args.baselines:
        plot_baselines(args.baselines, table, args.out)

    columns = ["run", "group", "n_complexes", "n_scored", "top1_le_2A_pct", "bestpose_le_2A_pct", "median_rmsd_top1", "median_rmsd_bestpose"]
    lines = ["Pose prediction (success = RMSD <= 2 A; denominator = all complexes, failed runs count as failures)",
             table[columns].round(2).to_string(index=False), "",
             "Top-N success at 2 A", topn.round(1).to_string(index=False), ""]
    if len(power):
        lines += ["Scoring power (positive correlation = lower energy goes with lower logKd, the expected direction)",
                  power.round(3).to_string(index=False), ""]
    for label, df in runs:
        counts = df.status.value_counts().to_dict()
        lines.append("%s status counts: %s" % (label, counts))
        if "oracle_state_top1_rmsd" in df and df.oracle_state_top1_rmsd.notna().any():
            lines.append("%s top-1 success with the best protonation state picked by RMSD (upper bound): %.1f%%" % (
                label, 100 * (df.oracle_state_top1_rmsd <= 2).sum() / len(df)))
    report = "\n".join(lines)
    with open(os.path.join(args.out, "report.txt"), "w") as fh:
        fh.write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
