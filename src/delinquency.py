"""Project 1 - Credit card delinquency prediction + "who do we call first?" prioritisation."""
from __future__ import annotations

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .utils import (MODELS, OUT, decile_table, ensure_dirs, inr, local_reasons,
                    make_preprocessor, recall_at_top, savefig, standard_metrics)

TARGET = "delinquent_90d"
BASE_NUM = ["income", "tenure_months", "credit_history_years", "credit_limit", "utilization",
            "balance", "late_12m", "payment_ratio", "debt_to_income", "num_products",
            "monthly_txns", "app_logins_30d"]
ENG_NUM = ["unpaid_share", "util_x_unpaid", "limit_to_income", "high_util_flag", "late_per_month_active"]
NUM = BASE_NUM + ENG_NUM
CAT = ["employment", "city_tier"]
# gender and age are deliberately NOT used as model inputs (see credit_scoring.py fairness audit)

CONTACT_COST = 600     # Rs per proactive outreach (agent time + payment-plan incentive)
SAVE_RATE = 0.08       # share of an at-risk balance we avoid losing when we reach out early


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["unpaid_share"] = 1 - d["payment_ratio"]
    d["util_x_unpaid"] = d["utilization"] * d["unpaid_share"]
    d["limit_to_income"] = d["credit_limit"] / d["income"]
    d["high_util_flag"] = (d["utilization"] > 0.8).astype(float)
    d["late_per_month_active"] = d["late_12m"] / np.clip(d["tenure_months"], 1, 12)
    d[NUM] = d[NUM].astype(float)
    return d


def _models(seed: int) -> dict:
    return {
        "Logistic Regression (class_weight=balanced)": LogisticRegression(
            max_iter=3000, class_weight="balanced"),
        "Gradient Boosting (no weighting)": HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.06, max_iter=300, random_state=seed),
        "Gradient Boosting (class_weight=balanced)": HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.06, max_iter=300, class_weight="balanced",
            random_state=seed),
    }


def _value_curve(y, p, balance, fractions):
    """Net rupee value of contacting the top-k% riskiest customers, model vs random."""
    order = np.argsort(-p)
    total_bad_balance = (y * balance).sum()
    rows = []
    for f in fractions:
        k = int(len(y) * f)
        idx = order[:k]
        model_val = (y[idx] * balance[idx]).sum() * SAVE_RATE - k * CONTACT_COST
        random_val = f * total_bad_balance * SAVE_RATE - k * CONTACT_COST
        rows.append((f, model_val, random_val))
    return pd.DataFrame(rows, columns=["contact_fraction", "model_value", "random_value"])


def run(df: pd.DataFrame, seed: int = 42, save: bool = True) -> dict:
    ensure_dirs()
    d = engineer(df)
    X, y = d[NUM + CAT], d[TARGET].values
    Xtr, Xte, ytr, yte, _, idx_te = train_test_split(
        X, y, d.index, test_size=0.25, stratify=y, random_state=seed)
    bal_te = d.loc[idx_te, "balance"].values

    results, fitted = {}, {}
    for name, model in _models(seed).items():
        pipe = Pipeline([("pre", make_preprocessor(NUM, CAT)), ("model", model)])
        pipe.fit(Xtr, ytr)
        p = pipe.predict_proba(Xte)[:, 1]
        fitted[name] = (pipe, p)
        m = standard_metrics(yte, p)
        m["precision@0.5"] = round(float(precision_score(yte, p >= 0.5, zero_division=0)), 4)
        m["recall@0.5"] = round(float(recall_score(yte, p >= 0.5)), 4)
        m["recall_in_top_20pct"] = round(recall_at_top(yte, p, 0.20), 4)
        results[name] = m

    champion = max(results, key=lambda k: results[k]["pr_auc"])
    base_pipe = fitted[champion][0]

    # class weighting inflates probabilities, so calibrate the champion on held-out data
    # (needed because the business rule below compares expected loss with a rupee cost)
    Xfit, Xcal, yfit, ycal = train_test_split(Xtr, ytr, test_size=0.2, stratify=ytr, random_state=seed)
    pipe = CalibratedClassifierCV(FrozenEstimator(clone(base_pipe).fit(Xfit, yfit)), method="sigmoid")
    pipe.fit(Xcal, ycal)
    p = pipe.predict_proba(Xte)[:, 1]
    calibrated = standard_metrics(yte, p)
    calibrated["mean_predicted_rate"] = round(float(p.mean()), 4)
    calibrated["actual_rate"] = round(float(yte.mean()), 4)

    # ---- business framing: how many customers should the collections team call? ----------
    fractions = np.arange(0.05, 1.01, 0.05)
    val = _value_curve(yte, p, bal_te, fractions)
    best = val.loc[val["model_value"].idxmax()]
    # per-customer rule: contact when expected avoided loss > cost of contact
    ev = p * bal_te * SAVE_RATE - CONTACT_COST
    chosen = ev > 0
    k = int(chosen.sum())
    rule_value = float((yte[chosen] * bal_te[chosen]).sum() * SAVE_RATE - k * CONTACT_COST)
    rand_value = float(k / len(yte) * (yte * bal_te).sum() * SAVE_RATE - k * CONTACT_COST)
    deciles = decile_table(yte, p)
    deciles.to_csv(OUT / "delinquency_deciles.csv", index=False)

    # ---- explainability --------------------------------------------------------------------
    rng_idx = np.random.default_rng(seed).choice(len(Xte), size=min(3000, len(Xte)), replace=False)
    imp = permutation_importance(pipe, Xte.iloc[rng_idx], yte[rng_idx], scoring="average_precision",
                                 n_repeats=5, random_state=seed)
    imp_df = (pd.DataFrame({"feature": NUM + CAT, "importance": imp.importances_mean})
              .sort_values("importance", ascending=False))
    imp_df.to_csv(OUT / "delinquency_feature_importance.csv", index=False)

    ref = {**Xtr[NUM].median().to_dict(), **Xtr[CAT].mode().iloc[0].to_dict()}
    top_idx = np.argsort(-p)[:25]
    top = Xte.iloc[top_idx]
    contact = pd.DataFrame({
        "customer_id": d.loc[idx_te, "customer_id"].values[top_idx],
        "default_probability": p[top_idx].round(3),
        "card_balance": bal_te[top_idx],
        "actually_delinquent": yte[top_idx],
        "top_reasons": local_reasons(pipe, top, ref, NUM + CAT),
    })
    contact.to_csv(OUT / "delinquency_priority_contact_list.csv", index=False)

    shap_note = _try_shap(pipe, Xte.iloc[rng_idx[:300]], Xtr.iloc[:200])

    # ---- plots ----------------------------------------------------------------------------
    figs = []
    plt.figure(figsize=(6.2, 4.6))
    for name, (_, pp) in fitted.items():
        pr, rc, _ = precision_recall_curve(yte, pp)
        plt.plot(rc, pr, label=f"{name.split(' (')[0]} ({name.split('(')[1][:-1]})")
    plt.axhline(yte.mean(), ls="--", c="grey", label="No-skill")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Delinquency: precision-recall")
    plt.legend(fontsize=7)
    figs.append(savefig("delinquency_pr_curve.png"))

    plt.figure(figsize=(6.2, 4.6))
    plt.bar(deciles["decile"], deciles["bad_rate"] * 100, color="#c0392b")
    plt.axhline(yte.mean() * 100, ls="--", c="grey")
    plt.xlabel("Risk decile (1 = riskiest)"); plt.ylabel("Actual delinquency rate (%)")
    plt.title("Delinquency rate by model risk decile")
    figs.append(savefig("delinquency_deciles.png"))

    plt.figure(figsize=(6.2, 4.6))
    plt.plot(val["contact_fraction"] * 100, val["model_value"] / 1e5, marker="o", label="Model-ranked")
    plt.plot(val["contact_fraction"] * 100, val["random_value"] / 1e5, marker="o", label="Random")
    plt.xlabel("% of customers contacted"); plt.ylabel("Net value (Rs lakh)")
    plt.title("Value of early outreach"); plt.legend()
    figs.append(savefig("delinquency_value_curve.png"))

    top_imp = imp_df.head(10).iloc[::-1]
    plt.figure(figsize=(6.2, 4.6))
    plt.barh(top_imp["feature"], top_imp["importance"], color="#2c7fb8")
    plt.xlabel("Drop in PR-AUC when shuffled"); plt.title("Top drivers (permutation importance)")
    figs.append(savefig("delinquency_importance.png"))

    if save:
        joblib.dump({"pipe": pipe, "num": NUM, "cat": CAT, "ref": ref, "champion": champion},
                    MODELS / "delinquency.joblib")

    return {
        "champion": champion,
        "models": results,
        "champion_calibrated": calibrated,
        "base_rate": round(float(yte.mean()), 4),
        "ev_rule_contact_share": round(k / len(yte), 3),
        "ev_rule_net_value_model": inr(rule_value),
        "ev_rule_net_value_random_same_effort": inr(rand_value),
        "best_fraction_on_grid": round(float(best["contact_fraction"]), 2),
        "top_decile_lift": round(float(deciles.loc[0, "lift"]), 2),
        "top_drivers": imp_df.head(5)["feature"].tolist(),
        "shap": shap_note,
        "figures": figs,
    }


def _try_shap(pipe, X_explain, X_background):
    """Optional SHAP summary plot (only runs if `pip install shap` is available)."""
    try:
        import shap
    except Exception:
        return "shap not installed - used permutation importance + occlusion reasons instead"
    try:
        pre, model = pipe.named_steps["pre"], pipe.named_steps["model"]
        names = list(pre.get_feature_names_out())
        Xb = pre.transform(X_background)
        Xe = pre.transform(X_explain)
        f = (lambda a: model.predict_proba(a)[:, 1])
        explainer = shap.Explainer(f, Xb, feature_names=names)
        sv = explainer(Xe[:150])
        shap.summary_plot(sv, Xe[:150], feature_names=names, show=False)
        return "SHAP plot saved: " + savefig("delinquency_shap_summary.png")
    except Exception as exc:  # never let an optional plot break the pipeline
        plt.close("all")
        return f"shap failed ({type(exc).__name__}); permutation importance used instead"
