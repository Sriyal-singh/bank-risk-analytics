"""Streamlit demo:  streamlit run app.py   (run `python run_all.py` once first to train models)."""
import streamlit as st

from src import predict

st.set_page_config(page_title="Bank Risk Analytics", layout="wide")
st.title("Bank Risk Analytics - live demo")
st.caption("Four models trained on synthetic data. Change the inputs to see predictions update.")

t1, t2, t3, t4 = st.tabs(["Card delinquency", "Loan approval", "Fraud check", "Churn & retention"])

with t1:
    c = st.columns(3)
    income = c[0].number_input("Annual income (Rs)", 150_000, 6_000_000, 600_000, 50_000, key="d_inc")
    limit = c[0].number_input("Credit limit (Rs)", 20_000, 2_000_000, 200_000, 10_000)
    util = c[1].slider("Utilisation", 0.0, 1.0, 0.6, 0.01)
    late = c[1].slider("Late payments (12m)", 0, 10, 2)
    pay = c[2].slider("Share of statement paid", 0.0, 1.0, 0.4, 0.01)
    dti = c[2].slider("Debt-to-income", 0.0, 0.9, 0.35, 0.01)
    r = predict.delinquency_risk({"income": float(income), "credit_limit": float(limit), "utilization": util,
                                  "late_12m": late, "payment_ratio": pay, "debt_to_income": dti})
    st.metric("Probability of 90-day delinquency", f"{r['probability']:.1%}")
    st.write("**Contact proactively?**", "Yes" if r["contact"] else "No - expected benefit below contact cost")
    st.write("**Main reasons:**", r["reasons"] or "no single dominant driver")

with t2:
    c = st.columns(3)
    income2 = c[0].number_input("Annual income (Rs)", 150_000, 6_000_000, 500_000, 50_000, key="l_inc")
    loan = c[0].number_input("Loan amount (Rs)", 50_000, 10_000_000, 600_000, 50_000)
    term = c[1].selectbox("Term (months)", [12, 24, 36, 48, 60, 84], 2)
    emp = c[1].selectbox("Employment", ["Salaried", "Self-employed", "Business", "Other"])
    dti2 = c[2].slider("Debt-to-income", 0.0, 0.9, 0.3, 0.01, key="l_dti")
    late2 = c[2].slider("Late payments (12m)", 0, 10, 0, key="l_late")
    r = predict.credit_decision({"income": float(income2), "loan_amount": float(loan), "loan_term_months": term,
                                 "employment": emp, "debt_to_income": dti2, "late_12m": late2})
    st.metric("Credit score", f"{r['score']:.0f}", f"cut-off {r['cutoff']:.0f}")
    st.write("**Decision:**", "APPROVE" if r["approved"] else "DECLINE", f"(estimated default risk {r['prob_default']:.1%})")

with t3:
    c = st.columns(3)
    amt = c[0].number_input("Amount (Rs)", 10, 2_000_000, 45_000)
    hour = c[0].slider("Hour of day", 0, 23, 3)
    cat = c[1].selectbox("Merchant", ["grocery", "fuel", "restaurant", "electronics", "travel",
                                      "online_retail", "atm", "jewellery"], 3)
    foreign = c[1].checkbox("Foreign transaction", True)
    cp = c[1].checkbox("Card present", False)
    vel = c[2].slider("Transactions in last hour", 1, 8, 3)
    dev = c[2].checkbox("New device", True)
    dist = c[2].number_input("Distance from home (km)", 0, 10000, 1200)
    r = predict.fraud_risk({"amount": float(amt), "hour": hour, "merchant_category": cat, "is_foreign": int(foreign),
                            "card_present": int(cp), "txn_count_1h": vel, "new_device": int(dev),
                            "distance_from_home_km": float(dist), "amount_ratio": amt / 1000.0})
    st.metric("Fraud risk score", f"{r['score']:.2f}", f"alert threshold {r['threshold']:.2f}")
    st.write("**Action:**", "BLOCK / REVIEW" if r["alert"] else "Allow")

with t4:
    c = st.columns(3)
    inactive = c[0].slider("Inactive months", 0, 6, 3)
    logins = c[0].slider("App logins (30d)", 0, 30, 2)
    comp = c[1].slider("Complaints (12m)", 0, 6, 1)
    salary = c[1].checkbox("Salary credited to this bank", False)
    prods = c[2].slider("Products held", 1, 6, 1)
    rev = c[2].number_input("Annual revenue from customer (Rs)", 1_000, 200_000, 15_000, 1_000)
    r = predict.churn_risk({"inactive_months": inactive, "app_logins_30d": logins, "complaints_12m": comp,
                            "salary_credited": int(salary), "num_products": prods, "annual_revenue": float(rev)})
    st.metric("Churn probability", f"{r['probability']:.1%}")
    st.write("**Expected net value of an offer:**", f"Rs {r['expected_net_value']:,.0f}")
    st.write("**Recommended action:**", r["action"])
