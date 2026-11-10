"""Project 3 - Card fraud detection: extreme imbalance, anomaly detection vs supervised,
time-based validation and a cost-based alert threshold."""
from __future__ import annotations

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline

from .utils import MODELS, ensure_dirs, inr, make_preprocessor, savefig

TARGET = "is_fraud"
NUM = ["amount", "hour", "is_foreign", "card_present", "txn_count_1h", "distance_from_home_km",
       "new_device", "amount_ratio", "log_amount", "is_night"]
CAT = ["merchant_category"]
REVIEW_COST = 120  # Rs per alert reviewed by an analyst / customer called


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["log_amount"] = np.log1p(d["amount"])
    d["is_night"] = (d["hour"] <= 5).astype(int)
    d[NUM] = d[NUM].astype(float)
    return d


def _cost(y, amount, alert) -> float:
    missed = (y == 1) & (~alert)
    return float(amount[missed].sum() + REVIEW_COST * alert.sum())


def run(tx: pd.DataFrame, seed: int = 42, save: bool = True) -> dict:
    ensure_dirs()
    d = engineer(tx.sort_values("timestamp").reset_index(drop=True))
    n = len(d)
    i1, i2 = int(n * 0.6), int(n * 0.8)  # time-ordered split: train / validation / test
    tr, va, te = d.iloc[:i1], d.iloc[i1:i2], d.iloc[i2:]
    Xtr, Xva, Xte = tr[NUM + CAT], va[NUM + CAT], te[NUM + CAT]
    ytr, yva, yte = tr[TARGET].values, va[TARGET].values, te[TARGET].values
    pos_w = float(np.sqrt((ytr == 0).sum() / max(ytr.sum(), 1)))  # softened class weight

    models = {
        "Isolation Forest (unsupervised)": Pipeline([
            ("pre", make_preprocessor(NUM, CAT)),
            ("m", IsolationForest(n_estimators=200, contamination="auto", random_state=seed, n_jobs=-1))]),
        "Logistic Regression (balanced)": Pipeline([
            ("pre", make_preprocessor(NUM, CAT)),
            ("m", LogisticRegression(max_iter=3000, class_weight="balanced"))]),
        "Gradient Boosting (weighted)": Pipeline([
            ("pre", make_preprocessor(NUM, CAT, scale=False)),
            ("m", HistGradientBoostingClassifier(max_depth=5, learning_rate=0.06, max_iter=250,
                                                  class_weight={0: 1.0, 1: pos_w}, random_state=seed))]),
    }

    def scorer(name, pipe, X):
        if name.startswith("Isolation"):
            return -pipe.named_steps["m"].score_samples(pipe.named_steps["pre"].transform(X))
        return pipe.predict_proba(X)[:, 1]

    scores_va, scores_te, results = {}, {}, {}
    for name, pipe in models.items():
        pipe.fit(Xtr, ytr) if not name.startswith("Isolation") else pipe.fit(Xtr)
        scores_va[name], scores_te[name] = scorer(name, pipe, Xva), scorer(name, pipe, Xte)
        s = scores_te[name]
        order = np.argsort(-s)
        k = max(1, int(0.01 * len(s)))  # review the riskiest 1% of transactions
        results[name] = {
            "pr_auc": round(float(average_precision_score(yte, s)), 4),
            "roc_auc": round(float(roc_auc_score(yte, s)), 4),
            "precision_at_top_1pct": round(float(yte[order[:k]].mean()), 4),
            "recall_at_top_1pct": round(float(yte[order[:k]].sum() / max(yte.sum(), 1)), 4),
        }
    results["Naive baseline: always predict 'not fraud'"] = {
        "accuracy": round(float(1 - yte.mean()), 4), "recall": 0.0,
        "note": "looks great on accuracy, catches zero fraud"}

    champion = max(scores_va, key=lambda k: average_precision_score(yva, scores_va[k]))

    # cost-based threshold chosen on VALIDATION only, then evaluated on the untouched TEST set
    s_va, s_te = scores_va[champion], scores_te[champion]
    cand = np.unique(np.quantile(s_va, np.linspace(0.90, 0.9995, 120)))
    costs = [_cost(yva, va["amount"].values, s_va >= t) for t in cand]
    thr = float(cand[int(np.argmin(costs))])
    alert = s_te >= thr
    amt = te["amount"].values
    fraud_loss_no_model = float(amt[yte == 1].sum())
    cost_with_model = _cost(yte, amt, alert)
    tp = int(((yte == 1) & alert).sum())
    business = {
        "alert_threshold": round(thr, 4),
        "alerts_per_10k_transactions": round(float(alert.mean() * 10_000), 1),
        "precision": round(tp / max(int(alert.sum()), 1), 4),
        "recall": round(tp / max(int(yte.sum()), 1), 4),
        "fraud_loss_without_model": inr(fraud_loss_no_model),
        "total_cost_with_model (missed fraud + review cost)": inr(cost_with_model),
        "net_saving": inr(fraud_loss_no_model - cost_with_model),
        "saving_pct": round(100 * (1 - cost_with_model / fraud_loss_no_model), 1),
    }

    figs = []
    plt.figure(figsize=(6.2, 4.6))
    for name, s in scores_te.items():
        pr, rc, _ = precision_recall_curve(yte, s)
        plt.plot(rc, pr, label=f"{name.split(' (')[0]} (AP={results[name]['pr_auc']:.2f})")
    plt.axhline(yte.mean(), ls="--", c="grey", label="No-skill")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Fraud: precision-recall (test)")
    plt.legend(fontsize=7)
    figs.append(savefig("fraud_pr_curve.png"))

    plt.figure(figsize=(6.2, 4.6))
    plt.plot(cand, np.array(costs) / 1e3, c="#8e44ad")
    plt.axvline(thr, ls="--", c="k", label="chosen threshold")
    plt.xlabel("Alert threshold (risk score)"); plt.ylabel("Total cost on validation (Rs '000)")
    plt.title("Picking the alert threshold by cost"); plt.legend()
    figs.append(savefig("fraud_cost_curve.png"))

    if save:
        joblib.dump({"pipe": models[champion], "name": champion, "threshold": thr,
                     "num": NUM, "cat": CAT}, MODELS / "fraud.joblib")

    return {
        "fraud_rate_test": round(float(yte.mean()), 5),
        "models_test": results, "champion": champion,
        "business": business, "figures": figs,
    }
