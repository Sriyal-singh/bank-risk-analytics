"""Project 4 - Bank customer churn: who is leaving, who is worth saving, and what is it worth."""
from __future__ import annotations

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .utils import (MODELS, OUT, decile_table, ensure_dirs, inr, make_preprocessor, savefig,
                    standard_metrics)

TARGET = "churned"
NUM = ["tenure_months", "num_products", "avg_account_balance", "monthly_txns", "app_logins_30d",
       "complaints_12m", "inactive_months", "salary_credited", "income", "age"]
CAT = ["employment", "city_tier"]

OFFER_COST = 1500       # Rs per retention offer (RM call + incentive)
EFFECTIVENESS = 0.25    # share of would-be churners the offer actually keeps
VALUE_YEARS = 2         # years of revenue protected when a customer is saved


def next_best_action(row: pd.Series) -> str:
    """Rule-based playbook mapping the main churn signal to a retention action."""
    if row["complaints_12m"] >= 2:
        return "Service-recovery call from relationship manager"
    if row["inactive_months"] >= 3:
        return "Re-engagement offer (cashback on next 5 transactions)"
    if row["salary_credited"] == 0:
        return "Salary-account upgrade offer"
    if row["num_products"] <= 1:
        return "Cross-sell bundle (card + savings)"
    return "Loyalty reward / fee waiver"


def run(df: pd.DataFrame, seed: int = 42, save: bool = True) -> dict:
    ensure_dirs()
    X = df[NUM + CAT].copy()
    X[NUM] = X[NUM].astype(float)
    y = df[TARGET].values
    Xtr, Xte, ytr, yte, _, idx_te = train_test_split(X, y, df.index, test_size=0.3, stratify=y, random_state=seed)
    rev_te = df.loc[idx_te, "annual_revenue"].values

    cands = {
        "Logistic Regression": Pipeline([("pre", make_preprocessor(NUM, CAT)),
                                         ("m", LogisticRegression(max_iter=3000))]),
        "Gradient Boosting": Pipeline([("pre", make_preprocessor(NUM, CAT)),
                                       ("m", HistGradientBoostingClassifier(
                                           max_depth=4, learning_rate=0.06, max_iter=250, random_state=seed))]),
    }
    res, probs = {}, {}
    for name, pipe in cands.items():
        pipe.fit(Xtr, ytr)
        probs[name] = pipe.predict_proba(Xte)[:, 1]
        res[name] = standard_metrics(yte, probs[name])
    champion = max(res, key=lambda k: res[k]["pr_auc"])

    # calibrated probabilities are required for expected-value targeting
    Xfit, Xcal, yfit, ycal = train_test_split(Xtr, ytr, test_size=0.2, stratify=ytr, random_state=seed)
    pipe = CalibratedClassifierCV(FrozenEstimator(clone(cands[champion]).fit(Xfit, yfit)), method="sigmoid")
    pipe.fit(Xcal, ycal)
    p = pipe.predict_proba(Xte)[:, 1]

    # ---- retention economics -----------------------------------------------------------------
    value_at_stake = rev_te * VALUE_YEARS
    ev = p * EFFECTIVENESS * value_at_stake - OFFER_COST     # expected net value of making an offer
    target = ev > 0
    n_t = int(target.sum())

    def realised(mask):
        """Estimated net value: offers keep EFFECTIVENESS of actual churners in the group."""
        return float((yte[mask] * EFFECTIVENESS * value_at_stake[mask]).sum() - OFFER_COST * mask.sum())

    rng = np.random.default_rng(seed)
    random_mask = np.zeros(len(yte), bool)
    random_mask[rng.choice(len(yte), n_t, replace=False)] = True
    strategy = {
        "customers_in_test_set": int(len(yte)),
        "actual_churn_rate": round(float(yte.mean()), 4),
        "revenue_at_risk (annual, churners)": inr(float((yte * rev_te).sum())),
        "model_targets": n_t,
        "model_targets_pct": round(100 * n_t / len(yte), 1),
        "churners_reached_pct": round(100 * float(yte[target].sum() / max(yte.sum(), 1)), 1),
        "net_value_model_targeting": inr(realised(target)),
        "net_value_random_same_size": inr(realised(random_mask)),
        "net_value_offer_everyone": inr(realised(np.ones(len(yte), bool))),
        "assumptions": f"offer cost Rs {OFFER_COST}, {int(EFFECTIVENESS * 100)}% of churners retained, "
                       f"{VALUE_YEARS} years of revenue protected",
    }

    # next-best-action playbook for targeted customers
    tgt = Xte[target].copy()
    tgt["churn_probability"] = p[target]
    tgt["annual_revenue"] = rev_te[target]
    tgt["expected_net_value"] = ev[target]
    tgt["action"] = tgt.apply(next_best_action, axis=1)
    tgt.insert(0, "customer_id", df.loc[idx_te[target], "customer_id"].values)
    tgt.sort_values("expected_net_value", ascending=False).to_csv(OUT / "churn_retention_targets.csv", index=False)
    actions = tgt["action"].value_counts().to_dict()

    # drivers
    sub = np.random.default_rng(seed).choice(len(Xte), size=min(3000, len(Xte)), replace=False)
    imp = permutation_importance(pipe, Xte.iloc[sub], yte[sub], scoring="average_precision",
                                 n_repeats=5, random_state=seed)
    imp_df = pd.DataFrame({"feature": NUM + CAT, "importance": imp.importances_mean}).sort_values(
        "importance", ascending=False)
    imp_df.to_csv(OUT / "churn_feature_importance.csv", index=False)
    deciles = decile_table(yte, p)
    deciles.to_csv(OUT / "churn_deciles.csv", index=False)

    # net value as we widen the campaign (customers sorted by expected value)
    order = np.argsort(-ev)
    cum = np.cumsum((yte[order] * EFFECTIVENESS * value_at_stake[order]) - OFFER_COST)
    figs = []
    plt.figure(figsize=(6.2, 4.6))
    plt.plot(np.arange(1, len(cum) + 1) / len(cum) * 100, cum / 1e5, c="#16a085")
    plt.axvline(100 * n_t / len(yte), ls="--", c="k", label="EV > 0 cut-off")
    plt.xlabel("% of customers offered (sorted by expected value)"); plt.ylabel("Cumulative net value (Rs lakh)")
    plt.title("Retention campaign: how far to go"); plt.legend()
    figs.append(savefig("churn_campaign_value.png"))

    top = imp_df.head(8).iloc[::-1]
    plt.figure(figsize=(6.2, 4.6))
    plt.barh(top["feature"], top["importance"], color="#e67e22")
    plt.xlabel("Drop in PR-AUC when shuffled"); plt.title("What drives churn")
    figs.append(savefig("churn_drivers.png"))

    if save:
        joblib.dump({"pipe": pipe, "num": NUM, "cat": CAT}, MODELS / "churn.joblib")

    return {
        "champion": champion, "models": res, "strategy": strategy,
        "action_playbook_counts": actions, "top_drivers": imp_df.head(5)["feature"].tolist(),
        "top_decile_lift": round(float(deciles.loc[0, "lift"]), 2), "figures": figs,
    }
