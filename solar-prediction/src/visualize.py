"""The paper's five diagnostic plots, saved under results/plots/."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

PLOTS = config.RESULTS_DIR / "plots"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def _save(fig, name):
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(PLOTS / name, dpi=130, bbox_inches="tight")
    plt.close(fig)


def line_actual_vs_pred(dates, actual, pred, model_name):
    # break the lines where the data has a gap (> 2 days) so the May-Jul 2026 gap
    # isn't drawn as a straight diagonal line
    d = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    a = pd.Series(actual, dtype=float).copy()
    p = pd.Series(pred, dtype=float).copy()
    gap = d.diff().dt.days.fillna(1) > 2
    a[gap] = np.nan
    p[gap] = np.nan
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(d, a, color=BLUE, lw=1.8, label="Actual")
    ax.plot(d, p, color=ORANGE, lw=1.8, ls="--", label="Predicted")
    ax.set_title(f"2026 Actual vs Predicted solar generation — {model_name}")
    ax.set_ylabel("Solar generation (MU)"); ax.set_xlabel("Date")
    ax.legend(); ax.grid(alpha=0.25)
    _save(fig, "1_actual_vs_predicted_line.png")


def scatter_actual_vs_pred(actual, pred, model_name):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(actual, pred, s=18, color=BLUE, alpha=0.6, edgecolor="white", linewidth=0.3)
    lo, hi = min(actual.min(), pred.min()), max(actual.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], color="#888", lw=1, ls="--", label="Perfect prediction")
    ax.set_title(f"Actual vs Predicted — {model_name}")
    ax.set_xlabel("Actual (MU)"); ax.set_ylabel("Predicted (MU)")
    ax.legend(); ax.grid(alpha=0.25)
    _save(fig, "2_actual_vs_predicted_scatter.png")


def residual_plot(actual, pred, model_name):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(actual, actual - pred, s=16, color=ORANGE, alpha=0.6, edgecolor="white", linewidth=0.3)
    ax.axhline(0, color="#888", lw=1)
    ax.set_title(f"Residuals (actual − predicted) — {model_name}")
    ax.set_xlabel("Actual (MU)"); ax.set_ylabel("Residual (MU)")
    ax.grid(alpha=0.25)
    _save(fig, "3_residuals.png")


def monthly_box(dates, actual, pred, model_name):
    df = pd.DataFrame({"month": pd.to_datetime(dates).dt.strftime("%b"),
                       "mnum": pd.to_datetime(dates).dt.month, "actual": actual, "pred": pred})
    order = df.sort_values("mnum")["month"].unique()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    pos = np.arange(len(order))
    a_data = [df[df.month == m]["actual"].values for m in order]
    p_data = [df[df.month == m]["pred"].values for m in order]
    b1 = ax.boxplot(a_data, positions=pos - 0.19, widths=0.34, patch_artist=True)
    b2 = ax.boxplot(p_data, positions=pos + 0.19, widths=0.34, patch_artist=True)
    for b in b1["boxes"]:
        b.set(facecolor=BLUE, alpha=0.55)
    for b in b2["boxes"]:
        b.set(facecolor=ORANGE, alpha=0.55)
    ax.set_xticks(pos); ax.set_xticklabels(order)
    ax.set_title(f"Monthly Actual vs Predicted — {model_name}")
    ax.set_ylabel("Solar generation (MU)")
    ax.legend([b1["boxes"][0], b2["boxes"][0]], ["Actual", "Predicted"])
    ax.grid(alpha=0.25, axis="y")
    _save(fig, "4_monthly_box.png")


def model_comparison_bars(metrics_df):
    """metrics_df: rows=model, columns include RMSE, MAE, R2 (with-module-temp run)."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric, color in zip(axes, ["RMSE", "MAE", "R2"], [BLUE, ORANGE, "#1baf7a"]):
        d = metrics_df.sort_values(metric, ascending=(metric != "R2"))
        ax.barh(d["model"], d[metric], color=color, alpha=0.8)
        ax.set_title(metric + (" (MU)" if metric in ("RMSE", "MAE") else ""))
        ax.grid(alpha=0.25, axis="x")
    fig.suptitle("Model comparison on 2026 test set (with module temperature)", y=1.03)
    _save(fig, "5_model_comparison.png")
