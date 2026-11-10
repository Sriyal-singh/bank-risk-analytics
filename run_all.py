"""Run all four projects end-to-end:   python run_all.py

Options
  --quick                      small data, runs in ~10 seconds (for a smoke test)
  --customers-csv PATH         use your own customer table instead of synthetic data
  --transactions-csv PATH      use your own transaction table instead of synthetic data
Your CSVs must contain the same column names as data/customers.csv and data/transactions.csv
(generate the synthetic ones first and copy the headers).
"""
from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from src import churn, credit_scoring, data, delinquency, fraud
from src.utils import DATA, OUT, ensure_dirs


def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--customers-csv")
    ap.add_argument("--transactions-csv")
    args = ap.parse_args()
    ensure_dirs()
    t0 = time.time()

    n_c, n_t = (6_000, 30_000) if args.quick else (25_000, 120_000)
    customers = (pd.read_csv(args.customers_csv) if args.customers_csv
                 else data.make_customers(n_c, args.seed))
    tx = (pd.read_csv(args.transactions_csv, parse_dates=["timestamp"]) if args.transactions_csv
          else data.make_transactions(n_t, args.seed + 1))
    if not args.customers_csv:
        customers.to_csv(DATA / "customers.csv", index=False)
    if not args.transactions_csv:
        tx.to_csv(DATA / "transactions.csv", index=False)

    results = {}
    for name, fn, frame in (
        ("1_delinquency", delinquency.run, customers),
        ("2_credit_scoring", credit_scoring.run, customers),
        ("3_fraud", fraud.run, tx),
        ("4_churn", churn.run, customers),
    ):
        print(f"[{time.time() - t0:5.1f}s] running {name} ...")
        results[name] = fn(frame, seed=args.seed)

    (OUT / "metrics.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nDone in {time.time() - t0:.1f}s. Results: outputs/metrics.json, figures: outputs/figures/")
    return results


if __name__ == "__main__":
    main()
