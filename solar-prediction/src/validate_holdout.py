"""
Hold-out validation — an independent check that the model really generalises.

The 2026 test already shows the model works on a future year. This script is a second,
stricter check on data we ALREADY have, with no capacity-share complications:

    train : 2024-04-01  ->  2025-06-30   (earlier data only)
    test  : 2025-07-01  ->  2025-12-31   (hidden; never seen in training OR scaling)

Because train and test sit in the same period/capacity regime, this isolates one
question: given a day's weather, can the model predict that day's solar generation?

Run:  .venv/bin/python src/validate_holdout.py
"""
import sys
import warnings
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))
import config
import dataset
import evaluate
from models import ann, ann_pso, decision_tree, gpr_model, linear, svm_model

warnings.filterwarnings("ignore")
MODELS = [linear, decision_tree, svm_model, gpr_model, ann, ann_pso]

SPLIT_DATE = "2025-07-01"      # everything before = train, from here to end of 2025 = test
TEST_END = "2026-01-01"


def run_holdout(models=None):
    """Train on the earlier period, score on the hidden later period.
    Returns (rows, info) where rows is a list of per-model metric dicts."""
    models = models or MODELS
    df = pd.read_csv(config.PROCESSED_DIR / "allindia_daily.csv", parse_dates=["date"])
    tr = df[df["date"] < SPLIT_DATE]
    te = df[(df["date"] >= SPLIT_DATE) & (df["date"] < TEST_END)]

    feats = dataset.FEATURES_WITH_TM
    # scalers fitted on TRAIN ONLY — the hidden period never influences anything
    xs = MinMaxScaler().fit(tr[feats].values)
    ys = MinMaxScaler().fit(tr[["capacity_factor"]].values)
    Xtr, Xte = xs.transform(tr[feats].values), xs.transform(te[feats].values)
    ytr = ys.transform(tr[["capacity_factor"]].values).ravel()
    actual = te[config.TARGET_COL].values
    cap_te = te["capacity_MW"].values

    rows = []
    for mod in models:
        est = mod.make()
        est.fit(Xtr, ytr)
        cf = ys.inverse_transform(est.predict(Xte).reshape(-1, 1)).ravel()
        pred = cf * cap_te                       # capacity factor -> real MU
        m = evaluate.metrics(actual, pred)
        rows.append({"model": mod.NAME, "Accuracy": 100 - m["MAPE"], **m})

    info = {
        "train_from": str(tr.date.min().date()), "train_to": str(tr.date.max().date()),
        "train_days": int(len(tr)),
        "test_from": str(te.date.min().date()), "test_to": str(te.date.max().date()),
        "test_days": int(len(te)), "actual_total_MU": float(actual.sum()),
    }
    return rows, info


def main():
    rows, info = run_holdout()
    print("=" * 72)
    print("HOLD-OUT VALIDATION — can the model predict days it has never seen?")
    print("=" * 72)
    print(f"  train : {info['train_from']} -> {info['train_to']}   ({info['train_days']} days)")
    print(f"  test  : {info['test_from']} -> {info['test_to']}   ({info['test_days']} days, hidden)")
    print()

    out = pd.DataFrame(rows).sort_values("RMSE")
    show = out[["model", "Accuracy", "MAPE", "RMSE", "MAE", "R2"]].copy()
    show["Accuracy"] = show["Accuracy"].map(lambda v: f"{v:.1f}%")
    show["MAPE"] = show["MAPE"].map(lambda v: f"{v:.2f}%")
    for c in ["RMSE", "MAE"]:
        show[c] = show[c].map(lambda v: f"{v:,.1f}")
    show["R2"] = show["R2"].map(lambda v: f"{v:.3f}")
    print(show.to_string(index=False))

    best = out.iloc[0]
    print(f"\nBest on hidden data: {best['model']} — {100 - best['MAPE']:.1f}% accurate "
          f"(MAPE {best['MAPE']:.2f}%, RMSE {best['RMSE']:.1f} MU)")
    print(f"Actual total over hidden period : {info['actual_total_MU']:,.0f} MU")
    out.to_csv(config.RESULTS_DIR / "holdout_validation.csv", index=False)
    print("Saved: results/holdout_validation.csv")


if __name__ == "__main__":
    main()
