"""
Phase 3+4+5 orchestrator + results export for the dashboard.

Trains all 6 models (Linear, Decision Tree, SVM, GPR, ANN, ANN-PSO) on 2024-25 and
tests on 2026 — once WITH module temperature and once WITHOUT (the paper's central
experiment). The model predicts a capacity factor (generation per MW installed), which
is converted back to real MU for every metric, so the growing panel fleet does not bias
the forecast.

CLI:  .venv/bin/python src/run.py      -> prints tables, saves metrics.csv, plots, JSON
Also exposes compute_results() / write_json() for the dashboard server.
"""
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))
import config
import dataset
import evaluate
import visualize
from models import ann, ann_pso, decision_tree, gpr_model, linear, svm_model

warnings.filterwarnings("ignore")
MODELS = [linear, decision_tree, svm_model, gpr_model, ann, ann_pso]
JSON_OUT = config.RESULTS_DIR / "dashboard_data.json"


def _run_one(module_temp: bool):
    data = dataset.load(with_module_temp=module_temp)
    rows, preds = [], {}
    for mod in MODELS:
        est = mod.make()
        est.fit(data.Xtr, data.ytr)
        pred = data.to_generation(est.predict(data.Xte))
        m = evaluate.metrics(data.yte_real, pred)
        rows.append({"model": mod.NAME, **m})
        preds[mod.NAME] = pred
    return pd.DataFrame(rows), preds, data


def compute_results() -> dict:
    """Train everything and return a JSON-serialisable results dict for the dashboard."""
    with_df, with_preds, data = _run_one(True)
    without_df, without_preds, _ = _run_one(False)

    def pack(df, preds):
        out = []
        for _, r in df.iterrows():
            name = r["model"]
            out.append({
                "model": name,
                "MSE": r["MSE"], "RMSE": r["RMSE"], "MAE": r["MAE"],
                "R2": r["R2"], "MAPE": r["MAPE"],
                "accuracy": 100.0 - r["MAPE"],              # intuitive accuracy = 100 - MAPE%
                "predicted_MU": [round(float(v), 1) for v in preds[name]],
            })
        return sorted(out, key=lambda d: d["RMSE"])

    best = min(with_df.to_dict("records"), key=lambda d: d["RMSE"])["model"]
    actual = [round(float(v), 1) for v in data.yte_real]
    best_pred = [round(float(v), 1) for v in with_preds[best]]

    # Independent hold-out check: train on earlier data, score on a hidden later period.
    # Only the production model is re-run here (keeps retraining fast).
    import validate_holdout
    prod_module = {m.NAME: m for m in MODELS}[best]
    hold_rows, hold_info = validate_holdout.run_holdout([prod_module])
    h = hold_rows[0]
    best_row = with_df.set_index("model").loc[best]
    validation = {
        "model": best,
        "holdout": {
            "period": f"{hold_info['test_from']} to {hold_info['test_to']}",
            "days": hold_info["test_days"],
            "train_days": hold_info["train_days"],
            "accuracy": round(h["Accuracy"], 1), "MAPE": round(h["MAPE"], 2),
            "RMSE": round(h["RMSE"], 1), "R2": round(h["R2"], 3),
        },
        "future_year": {
            "period": f"{data.dates_test.min().strftime('%Y-%m-%d')} to {data.dates_test.max().strftime('%Y-%m-%d')}",
            "days": len(actual),
            "accuracy": round(100 - float(best_row["MAPE"]), 1), "MAPE": round(float(best_row["MAPE"]), 2),
            "RMSE": round(float(best_row["RMSE"]), 1), "R2": round(float(best_row["R2"]), 3),
        },
    }
    return {
        "validation": validation,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dates": [d.strftime("%Y-%m-%d") for d in data.dates_test],
        "actual_MU": actual,
        "with_module_temp": pack(with_df, with_preds),
        "without_module_temp": pack(without_df, without_preds),
        "best_model": best,
        "summary": {
            "n_test_days": len(actual),
            "date_range": [data.dates_test.min().strftime("%Y-%m-%d"), data.dates_test.max().strftime("%Y-%m-%d")],
            "total_actual_MU": round(sum(actual), 1),
            "total_predicted_MU": round(sum(best_pred), 1),
            "best_rmse": round(float(min(r["RMSE"] for r in with_df.to_dict("records"))), 1),
            "best_mape": round(float(with_df.set_index("model").loc[best, "MAPE"]), 2),
            "best_r2": round(float(with_df.set_index("model").loc[best, "R2"]), 3),
        },
    }


def write_json(results: dict) -> Path:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(results, indent=2))
    return JSON_OUT


def _fmt(records):
    df = pd.DataFrame(records)[["model", "MSE", "RMSE", "MAE", "R2", "MAPE"]]
    for c in ["MSE", "RMSE", "MAE"]:
        df[c] = df[c].map(lambda v: f"{v:,.1f}")
    df["R2"] = df["R2"].map(lambda v: f"{v:.4f}")
    df["MAPE"] = df["MAPE"].map(lambda v: f"{v:.2f}%")
    return df.to_string(index=False)


def main():
    results = compute_results()
    write_json(results)

    # per-region models for the map view (same production model as the headline)
    import regions
    regions.write_json(regions.compute(results["best_model"]))

    print("=" * 70)
    print("RESULTS — predict 2026 daily solar generation (capacity-factor model)")
    print("=" * 70)
    print("\nWITH module temperature:")
    print(_fmt(results["with_module_temp"]))
    print("\nWITHOUT module temperature:")
    print(_fmt(results["without_module_temp"]))
    print(f"\nBest model: {results['best_model']}  "
          f"(RMSE {results['summary']['best_rmse']} MU, MAPE {results['summary']['best_mape']}%)")

    # plots for the best model
    data = dataset.load(with_module_temp=True)
    best = results["best_model"]
    best_pred = np.array(next(m["predicted_MU"] for m in results["with_module_temp"] if m["model"] == best))
    visualize.line_actual_vs_pred(data.dates_test, data.yte_real, best_pred, best)
    visualize.scatter_actual_vs_pred(data.yte_real, best_pred, best)
    visualize.residual_plot(data.yte_real, best_pred, best)
    visualize.monthly_box(data.dates_test, data.yte_real, best_pred, best)
    visualize.model_comparison_bars(pd.DataFrame(results["with_module_temp"]))
    pd.concat([pd.DataFrame(results["with_module_temp"]).assign(module_temp="with"),
               pd.DataFrame(results["without_module_temp"]).assign(module_temp="without")]
              ).to_csv(config.RESULTS_DIR / "metrics.csv", index=False)
    print(f"Saved: results/metrics.csv, results/dashboard_data.json, results/plots/*")


if __name__ == "__main__":
    main()
