from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr



# -----------------------------
# Config
# -----------------------------

@dataclass(frozen=True)
class EvalConfig:
    # LLM rating prefixes in your columns: zs_*, fs_*, pers_*
    llm_prefixes: Tuple[str, ...] = ("zs", "fs", "pers")

    # “Automated metric” columns that exist as standalone columns
    metric_cols: Tuple[str, ...] = ("ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore")

    # Human columns
    expert_prefix: str = "Expert_"
    crowd_prefix: str = "Crowd_"

    # Which correlation types to compute
    compute_pearson: bool = True
    compute_spearman: bool = True

    # Alpha for “significance tables” (for paper, you’ll likely replace this with FDR)
    alpha: float = 0.05


# -----------------------------
# IO + preprocessing
# -----------------------------

def load_jsonl(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_json(path, lines=True)
    return df


def coerce_likert_and_human_cols_to_float(df: pd.DataFrame, cfg: EvalConfig) -> pd.DataFrame:
    df = df.copy()

    prefixes = list(cfg.llm_prefixes) + [cfg.expert_prefix, cfg.crowd_prefix]
    cols = [c for c in df.columns if any(
        (c.startswith(p + "_") if p in cfg.llm_prefixes else c.startswith(p))
        for p in prefixes
    )]
    # include metric columns if present
    cols += [c for c in cfg.metric_cols if c in df.columns]
    cols = sorted(set(cols))

    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def infer_targets(df: pd.DataFrame, cfg: EvalConfig) -> List[str]:
    expert_targets = [c for c in df.columns if c.startswith(cfg.expert_prefix)]
    crowd_targets = [c for c in df.columns if c.startswith(cfg.crowd_prefix)]
    return expert_targets + crowd_targets


def infer_llm_cols(df: pd.DataFrame, cfg: EvalConfig) -> Dict[str, List[str]]:
    out = {}
    for p in cfg.llm_prefixes:
        out[p] = [c for c in df.columns if c.startswith(p + "_")]
    return out


# -----------------------------
# Core stats
# -----------------------------

def _safe_corr(x: np.ndarray, y: np.ndarray):
    """Return (pearson_r, pearson_p, spearman_r, spearman_p, n) with NaN safety."""
    m = np.isfinite(x) & np.isfinite(y)
    x2, y2 = x[m], y[m]
    n = len(x2)

    if n < 3 or np.std(x2) == 0 or np.std(y2) == 0:
        return (np.nan, np.nan, np.nan, np.nan, n)

    pr, pp = pearsonr(x2, y2)
    sr, sp = spearmanr(x2, y2)
    return (pr, pp, sr, sp, n)


def compute_correlations(df: pd.DataFrame, cfg: EvalConfig) -> pd.DataFrame:
    """
    Returns a long dataframe:
    LLM (source col), SourceGroup (zs/fs/pers/metric), Target, Pearson, Pearson_p, Spearman, Spearman_p, N
    """
    targets = infer_targets(df, cfg)
    llm_groups = infer_llm_cols(df, cfg)

    rows = []

    # 1) LLM prompt outputs (zs/fs/pers)
    for group, cols in llm_groups.items():
        for col in cols:
            x = df[col].to_numpy()
            for tgt in targets:
                y = df[tgt].to_numpy()
                pr, pp, sr, sp, n = _safe_corr(x, y)
                rows.append({
                    "SourceGroup": group,
                    "LLM": col,
                    "Target": tgt,
                    "Pearson": pr,
                    "Pearson_p": pp,
                    "Spearman": sr,
                    "Spearman_p": sp,
                    "N": n,
                })

    # 2) Automated metrics vs human targets
    for mcol in cfg.metric_cols:
        if mcol not in df.columns:
            continue
        x = df[mcol].to_numpy()
        for tgt in targets:
            y = df[tgt].to_numpy()
            pr, pp, sr, sp, n = _safe_corr(x, y)
            rows.append({
                "SourceGroup": "metric",
                "LLM": mcol,
                "Target": tgt,
                "Pearson": pr,
                "Pearson_p": pp,
                "Spearman": sr,
                "Spearman_p": sp,
                "N": n,
            })

    return pd.DataFrame(rows)


# -----------------------------
# Paper-ready tables
# -----------------------------

def top_pairs_table(corr_df: pd.DataFrame, kind: str = "Pearson", k: int = 10) -> pd.DataFrame:
    """
    kind: "Pearson" or "Spearman"
    """
    kind_p = f"{kind}_p"
    out = corr_df.dropna(subset=[kind]).sort_values(kind, ascending=False).head(k).copy()
    # Keep only useful columns (rename for readability)
    cols = ["LLM", "SourceGroup", "Target", kind, kind_p, "N"]
    out = out[cols]
    return out


def sig_pairs_table(corr_df: pd.DataFrame, kind: str, alpha: float) -> pd.DataFrame:
    kind_p = f"{kind}_p"
    out = corr_df[(corr_df[kind_p] < alpha) & corr_df[kind].notna()].sort_values(kind, ascending=False).copy()
    return out


def to_latex_booktabs(df: pd.DataFrame, path: str | Path, index: bool = False, caption: str = "", label: str = ""):
    latex = df.to_latex(index=index, escape=True, longtable=False, caption=caption, label=label, float_format="%.3f")
    # Light “booktabs-like” improvement (you can do more later)
    latex = latex.replace("\\toprule", "\\toprule").replace("\\midrule", "\\midrule").replace("\\bottomrule", "\\bottomrule")
    Path(path).write_text(latex, encoding="utf-8")


# -----------------------------
# Paper-ready figures
# -----------------------------

def plot_heatmap(corr_df: pd.DataFrame, value_col: str, outpath: str | Path, title: str):
    """
    Creates a heatmap with Source (LLM/metric col names) on y and Target on x.
    """
    import seaborn as sns  # keep seaborn here so rest can be pure matplotlib if you want

    pivot = corr_df.pivot(index="LLM", columns="Target", values=value_col)
    plt.figure(figsize=(14, max(6, 0.25 * len(pivot.index))))
    sns.heatmap(pivot, annot=False, center=0)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def compute_residuals_vs_expert(df: pd.DataFrame, cfg: EvalConfig) -> pd.DataFrame:
    """
    Residuals: |LLM_dim - Expert_dim| for each (zs/fs/pers column) where Expert_dim exists.
    Assumes naming like zs_fluency and Expert_Fluency (case sensitive).
    """
    rows = []
    expert_cols = [c for c in df.columns if c.startswith(cfg.expert_prefix)]
    # map Expert_Fluency -> "fluency"
    expert_dim_map = {c: c[len(cfg.expert_prefix):].lower() for c in expert_cols}
    dim_to_expertcol = {v: k for k, v in expert_dim_map.items()}

    for p in cfg.llm_prefixes:
        for col in [c for c in df.columns if c.startswith(p + "_")]:
            dim = col.split("_", 1)[1].lower()
            if dim not in dim_to_expertcol:
                continue
            ecol = dim_to_expertcol[dim]
            r = np.abs(df[col] - df[ecol])
            tmp = pd.DataFrame({
                "Prompt": p,
                "LLM": col,
                "Dimension": dim,
                "Residual": r
            })
            rows.append(tmp)

    if not rows:
        return pd.DataFrame(columns=["Prompt", "LLM", "Dimension", "Residual"])
    return pd.concat(rows, ignore_index=True)


def plot_residual_boxplot(resid_df: pd.DataFrame, outpath: str | Path, title: str):
    import seaborn as sns

    plt.figure(figsize=(14, 6))
    sns.boxplot(data=resid_df, x="Dimension", y="Residual", hue="Prompt")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

#more plots
from typing import Literal

def bootstrap_corr_ci(
    x: np.ndarray,
    y: np.ndarray,
    method: Literal["pearson", "spearman"] = "pearson",
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float, int]:
    """
    Returns (point_estimate, ci_low, ci_high, n_effective)
    Bootstraps over paired observations.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    n = len(x)

    if n < 3 or np.std(x) == 0 or np.std(y) == 0:
        return (np.nan, np.nan, np.nan, n)

    # point estimate
    if method == "pearson":
        point = pearsonr(x, y)[0]
    else:
        point = spearmanr(x, y)[0]

    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=float)

    idx = np.arange(n)
    for b in range(n_boot):
        samp = rng.choice(idx, size=n, replace=True)
        xb = x[samp]
        yb = y[samp]
        # guard against degenerate bootstrap samples
        if np.std(xb) == 0 or np.std(yb) == 0:
            stats[b] = np.nan
            continue
        if method == "pearson":
            stats[b] = pearsonr(xb, yb)[0]
        else:
            stats[b] = spearmanr(xb, yb)[0]

    stats = stats[np.isfinite(stats)]
    if len(stats) == 0:
        return (point, np.nan, np.nan, n)

    alpha = (1 - ci) / 2
    lo = np.quantile(stats, alpha)
    hi = np.quantile(stats, 1 - alpha)
    return (point, lo, hi, n)


def compute_prompt_expert_corrs_with_ci(
    df: pd.DataFrame,
    cfg: EvalConfig,
    method: Literal["pearson", "spearman"] = "pearson",
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Computes correlation (with bootstrap CI) between each prompt-dimension LLM score
    and the corresponding Expert dimension score.

    Output columns:
      Prompt, Dimension, LLM_col, Expert_col, r, ci_low, ci_high, N
    """
    df = df.copy()

    # Find expert columns and map them to dimension tokens
    expert_cols = [c for c in df.columns if c.startswith(cfg.expert_prefix)]
    # Example: Expert_Fluency -> "fluency"
    dim_to_expert = {c[len(cfg.expert_prefix):].strip().lower(): c for c in expert_cols}

    rows = []
    for prompt in cfg.llm_prefixes:
        llm_cols = [c for c in df.columns if c.startswith(prompt + "_")]
        for llm_col in llm_cols:
            # Example: zs_fluency -> "fluency"
            dim = llm_col.split("_", 1)[1].strip().lower()
            if dim not in dim_to_expert:
                continue

            expert_col = dim_to_expert[dim]
            x = df[llm_col].to_numpy()
            y = df[expert_col].to_numpy()

            r, lo, hi, n_eff = bootstrap_corr_ci(
                x, y, method=method, n_boot=n_boot, ci=ci, seed=seed
            )

            rows.append({
                "Prompt": prompt,
                "Dimension": dim,
                "LLM_col": llm_col,
                "Expert_col": expert_col,
                "r": r,
                "ci_low": lo,
                "ci_high": hi,
                "N": n_eff,
                "Method": method
            })

    return pd.DataFrame(rows)


def plot_forest_cross_model(
    ci_df: pd.DataFrame,
    outpath: str | Path,
    method: Literal["pearson", "spearman"] = "pearson",
    title: str | None = None,
):
    """
    Forest plot by dimension.
    - One panel per dimension
    - y-axis: models (repeated for each prompt)
    - x-axis: correlation r with bootstrap CI
    """
    ci_df = ci_df.copy()
    ci_df = ci_df[ci_df["Method"] == method].copy()

    # Stable ordering
    prompt_order = ["zs", "fs", "pers"]
    ci_df["Prompt"] = pd.Categorical(ci_df["Prompt"], categories=prompt_order, ordered=True)

    dim_order = ["fluency", "coherence", "consistency", "relevance"]
    dims_present = [d for d in dim_order if d in set(ci_df["Dimension"].unique())]
    if not dims_present:
        dims_present = sorted(ci_df["Dimension"].unique())

    models = sorted(ci_df["Model"].unique())

    # Layout: one subplot per dimension
    n_dims = len(dims_present)
    fig_h = max(6, 1.2 * len(models) + 1)
    fig, axes = plt.subplots(n_dims, 1, figsize=(12, fig_h * n_dims / 2), sharex=True)
    if n_dims == 1:
        axes = [axes]

    for ax, dim in zip(axes, dims_present):
        sub = ci_df[ci_df["Dimension"] == dim].copy()

        # Build y positions: for each model, allocate 3 rows (zs/fs/pers)
        y_positions = []
        y_labels = []
        points = []

        y = 0
        # Reverse so first model appears at top
        for model in models[::-1]:
            sub_m = sub[sub["Model"] == model].sort_values("Prompt")
            for p in prompt_order:
                row = sub_m[sub_m["Prompt"] == p]
                if len(row) == 0:
                    continue
                row = row.iloc[0]
                y_positions.append(y)
                y_labels.append(f"{model} · {p}")
                points.append(row)
                y += 1
            y += 0.5  # spacer between models

        # Plot
        xs = np.array([r["r"] for r in points], dtype=float)
        lo = np.array([r["ci_low"] for r in points], dtype=float)
        hi = np.array([r["ci_high"] for r in points], dtype=float)

        # Error bars (horizontal)
        xerr = np.vstack([xs - lo, hi - xs])
        ax.errorbar(xs, y_positions, xerr=xerr, fmt="o", capsize=3)

        ax.axvline(0.0, linewidth=1)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels, fontsize=9)
        ax.set_title(f"{dim.title()} — {method.title()} r (95% bootstrap CI)")

        # A little padding
        ax.set_ylim(min(y_positions) - 1, max(y_positions) + 1)

    axes[-1].set_xlabel("Correlation with Expert score")
    if title:
        fig.suptitle(title, y=0.995, fontsize=14)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

# PCA plots
def get_evaluator_columns(df: pd.DataFrame, cfg: EvalConfig) -> list[str]:
    """
    Evaluators = all LLM prompt rating columns + expert/crowd columns + metric columns (if present).
    """
    cols = []

    # LLM columns (zs_*, fs_*, pers_*)
    for p in cfg.llm_prefixes:
        cols.extend([c for c in df.columns if c.startswith(p + "_")])

    # Human columns
    cols.extend([c for c in df.columns if c.startswith(cfg.expert_prefix)])
    cols.extend([c for c in df.columns if c.startswith(cfg.crowd_prefix)])

    # Metrics
    cols.extend([c for c in cfg.metric_cols if c in df.columns])

    # Keep only numeric + drop duplicates
    cols = list(dict.fromkeys(cols))
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    return cols


def compute_evaluator_corr_matrix(df: pd.DataFrame, cfg: EvalConfig, method: str = "pearson") -> pd.DataFrame:
    """
    Returns correlation matrix across evaluators (columns vs columns).
    """
    cols = get_evaluator_columns(df, cfg)
    mat = df[cols].corr(method=method)
    return mat


def pca_kmeans_on_corr(
    corr: pd.DataFrame,
    n_components: int = 2,
    k: int = 3,
    standardize: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Treat each evaluator as a vector of correlations with all others (row in corr).
    PCA reduces to 2D, then KMeans clusters.
    Returns DataFrame with columns: evaluator, pc1, pc2, cluster
    """
    X = corr.fillna(0.0).to_numpy()

    if standardize:
        X = StandardScaler().fit_transform(X)

    pca = PCA(n_components=n_components, random_state=seed)
    Z = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_
    
    print("PC1 explained variance:", explained[0])
    print("PC2 explained variance:", explained[1])
    print("Total explained variance:", explained.sum())

    km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    labels = km.fit_predict(Z)

    out = pd.DataFrame({
        "evaluator": corr.index.to_list(),
        "pc1": Z[:, 0],
        "pc2": Z[:, 1],
        "cluster": labels
    })
    return out


def plot_pca_clusters(pca_df: pd.DataFrame, outpath: str | Path, title: str):
    """
    Scatterplot of PCA components with cluster labels and evaluator annotations.
    Uses matplotlib defaults (no manual colors needed).
    """
    plt.figure(figsize=(10, 6))

    # Plot per cluster (default color cycle)
    for cl in sorted(pca_df["cluster"].unique()):
        sub = pca_df[pca_df["cluster"] == cl]
        plt.scatter(sub["pc1"], sub["pc2"], label=f"cluster {cl}")

    # annotate (small font so it stays readable)
    for _, r in pca_df.iterrows():
        plt.text(r["pc1"], r["pc2"], r["evaluator"], fontsize=7)

    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


# -----------------------------
# Runner (multi-model)
# -----------------------------

def run_for_one_model(model_name: str, jsonl_path: str | Path, outdir: str | Path, cfg: EvalConfig) -> Dict[str, pd.DataFrame]:
    outdir = Path(outdir) / model_name
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_jsonl(jsonl_path)
    df = coerce_likert_and_human_cols_to_float(df, cfg)

        # --- NEW: PCA clustering on evaluator correlation matrix ---
    pearson_corr = compute_evaluator_corr_matrix(df, cfg, method="pearson")
    pearson_corr.to_csv(outdir / "evaluator_corr_matrix_pearson.csv")

    pca_df = pca_kmeans_on_corr(pearson_corr, k=3, seed=42)
    pca_df.to_csv(outdir / "pca_kmeans_clusters.csv", index=False)

    plot_pca_clusters(
        pca_df,
        outdir / "pca_kmeans_clusters.png",
        title=f"Evaluator clustering via PCA (Pearson corr) — {model_name}"
    )


        # --- NEW: bootstrap CIs for prompt-vs-expert correlations ---
    ci_pearson = compute_prompt_expert_corrs_with_ci(
        df, cfg, method="pearson", n_boot=2000, ci=0.95, seed=0
    )
    ci_spearman = compute_prompt_expert_corrs_with_ci(
        df, cfg, method="spearman", n_boot=2000, ci=0.95, seed=0
    )
    ci_df = pd.concat([ci_pearson, ci_spearman], ignore_index=True)
    ci_df.to_csv(outdir / "prompt_expert_corrs_with_ci.csv", index=False)

    corr_df = compute_correlations(df, cfg)
    corr_df.to_csv(outdir / "correlations_long.csv", index=False)

    # Tables
    top_p = top_pairs_table(corr_df, kind="Pearson", k=10)
    top_s = top_pairs_table(corr_df, kind="Spearman", k=10)
    top_p.to_csv(outdir / "top10_pearson.csv", index=False)
    top_s.to_csv(outdir / "top10_spearman.csv", index=False)

    to_latex_booktabs(top_p, outdir / "top10_pearson.tex", caption=f"Top 10 Pearson pairs ({model_name})", label=f"tab:{model_name}_top10_pearson")
    to_latex_booktabs(top_s, outdir / "top10_spearman.tex", caption=f"Top 10 Spearman pairs ({model_name})", label=f"tab:{model_name}_top10_spearman")

    sig_p = sig_pairs_table(corr_df, "Pearson", cfg.alpha)
    sig_s = sig_pairs_table(corr_df, "Spearman", cfg.alpha)
    sig_p.to_csv(outdir / "sig_pearson.csv", index=False)
    sig_s.to_csv(outdir / "sig_spearman.csv", index=False)

    # Figures
    plot_heatmap(corr_df, "Spearman", outdir / "heatmap_spearman.png", f"Spearman correlations ({model_name})")
    plot_heatmap(corr_df, "Pearson", outdir / "heatmap_pearson.png", f"Pearson correlations ({model_name})")

    resid_df = compute_residuals_vs_expert(df, cfg)
    resid_df.to_csv(outdir / "residuals_long.csv", index=False)
    if len(resid_df) > 0:
        plot_residual_boxplot(resid_df, outdir / "residuals_boxplot.png", f"|LLM − Expert| residuals ({model_name})")

    return {
        "corr": corr_df,
        "ci_df": ci_df,
        "pca_df": pca_df,
        "top_p": top_p,
        "top_s": top_s,
        "sig_p": sig_p,
        "sig_s": sig_s,
        "resid": resid_df,
    }


def run_all_models(model_files: Dict[str, str | Path], outdir: str | Path, cfg: Optional[EvalConfig] = None) -> pd.DataFrame:
    cfg = cfg or EvalConfig()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_corr = []
    all_ci = []
    for model_name, path in model_files.items():
        print(model_name)
        artifacts = run_for_one_model(model_name, path, outdir, cfg)
        c = artifacts["corr"].copy()
        c.insert(0, "Model", model_name)
        all_corr.append(c)

        ci = artifacts["ci_df"].copy()
        ci.insert(0, "Model", model_name)
        all_ci.append(ci)

    all_corr_df = pd.concat(all_corr, ignore_index=True)
    all_corr_df.to_csv(outdir / "ALL_MODELS_correlations_long.csv", index=False)

    # Cross-model summary table: best prompt-per-dimension vs Expert_* (Pearson)
    # You can expand this later, but it gives you a paper-ready “main result table”.
    expert_targets = sorted([t for t in all_corr_df["Target"].unique() if t.startswith(cfg.expert_prefix)])
    rows = []
    for model_name in all_corr_df["Model"].unique():
        sub = all_corr_df[all_corr_df["Model"] == model_name]
        for tgt in expert_targets:
            # best source correlation for that target
            ssub = sub[sub["Target"] == tgt].dropna(subset=["Pearson"])
            if len(ssub) == 0:
                continue
            best = ssub.sort_values("Pearson", ascending=False).iloc[0]
            rows.append({
                "Model": model_name,
                "Target": tgt,
                "BestSource": best["LLM"],
                "BestSourceGroup": best["SourceGroup"],
                "Pearson": best["Pearson"],
                "Spearman": best["Spearman"],
                "N": int(best["N"]),
            })

    summary = pd.DataFrame(rows).sort_values(["Target", "Pearson"], ascending=[True, False])
    summary.to_csv(outdir / "ALL_MODELS_best_by_target.csv", index=False)
    to_latex_booktabs(summary, outdir / "ALL_MODELS_best_by_target.tex",
                      caption="Best correlation (per model, per expert target)",
                      label="tab:all_models_best_by_target")

    return summary


#Example usage:
model_files = {
    "gpt-5.2": "gpt-5.2_merged_summ_eval.jsonl",
     "gpt-5": "gpt-5_merged_summ_eval.jsonl",
     "gpt-4.1-nano": "gpt-4.1-nano_merged_summ_eval.jsonl",
     "gemini-2.5-flash": "gemini_merged_summ_eval.jsonl",
     "gpt-5-nano": "gpt-5-nano_merged_summ_eval.jsonl",
     "o3-mini": "o3-mini_merged_summ_eval.jsonl",
     
 }
summary = run_all_models(model_files, outdir="outputs_multi_model")
print(summary.head())