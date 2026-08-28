"""
MPLAD Drishti - Real Data Anomaly Detection Engine
==================================================
Works with the real MPLADS data normalized by generate_data.py.

Supported from the supplied real datasets:
- expenditure vs recommended cost
- expenditure vs funds received
- under-utilization
- unreleased funds
- peer-relative state comparison
- Isolation Forest on financial features
- MP/state/district summaries
- explainable risk scoring

Not claimed when source fields are absent:
- vendor concentration
- delayed works
- duplicate works
- SC/ST allocation compliance
- physical asset verification
"""

from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"

LOW_UTILIZATION_THRESHOLD = 60.0
COST_OVERRUN_THRESHOLD = 130.0
UNRELEASED_SHARE_THRESHOLD = 50.0


def load_data():
    works = pd.read_csv(DATA_DIR / "mplads_works.csv")
    roster = pd.read_csv(DATA_DIR / "mp_roster.csv")
    return works, roster


def aggregate_financials(works, roster):
    g = works.groupby("mp_id", dropna=False).agg(
        mp_name=("mp_name", "first"),
        state=("state", "first"),
        constituency=("constituency", "first"),
        district=("district", "first"),
        house=("house", "first"),
        fund_received_cr=("fund_received_cr", "sum"),
        recommended_cost_cr=("sanctioned_amount_cr", "sum"),
        expenditure_cr=("final_expenditure_cr", "sum"),
        work_records=("work_id", "count"),
    ).reset_index()

    r = roster[[
        "mp_id", "entitlement_cr", "goi_release_cr", "unreleased_cr", "tenure"
    ]].drop_duplicates("mp_id")
    g = g.merge(r, on="mp_id", how="left")

    g["utilization_pct"] = np.where(
        g["fund_received_cr"] > 0,
        g["expenditure_cr"] / g["fund_received_cr"] * 100,
        0
    )
    g["cost_vs_recommended_pct"] = np.where(
        g["recommended_cost_cr"] > 0,
        g["expenditure_cr"] / g["recommended_cost_cr"] * 100,
        0
    )
    g["unreleased_share_pct"] = np.where(
        g["entitlement_cr"] > 0,
        g["unreleased_cr"] / g["entitlement_cr"] * 100,
        0
    )

    return g.fillna(0)


def run_ml(agg):
    features = [
        "fund_received_cr",
        "recommended_cost_cr",
        "expenditure_cr",
        "utilization_pct",
        "cost_vs_recommended_pct",
        "unreleased_share_pct",
        "work_records",
    ]
    X = agg[features].replace([np.inf, -np.inf], np.nan).fillna(0)

    if len(agg) < 8:
        agg["ml_anomaly"] = False
        agg["ml_anomaly_score"] = 0.0
        return agg

    model = IsolationForest(
        n_estimators=250,
        contamination=min(0.12, max(0.05, 5 / len(agg))),
        random_state=42
    )
    agg["ml_anomaly"] = model.fit_predict(X) == -1
    agg["ml_anomaly_score"] = (-model.decision_function(X)).round(4)
    return agg


def build_report(works, roster):
    agg = run_ml(aggregate_financials(works, roster))

    national_util = round(
        agg["utilization_pct"].mean(), 1
    ) if len(agg) else 0

    state_avg = agg.groupby("state")["utilization_pct"].mean().round(1).to_dict()

    report = []
    for _, r in agg.iterrows():
        reasons = []
        points = 0

        if r["cost_vs_recommended_pct"] >= COST_OVERRUN_THRESHOLD:
            excess = max(0, r["expenditure_cr"] - r["recommended_cost_cr"])
            reasons.append(
                f"Actual expenditure is {r['cost_vs_recommended_pct']:.1f}% of recommended work cost "
                f"(excess: ₹{excess:.2f} Cr)"
            )
            points += 2

        if r["utilization_pct"] < LOW_UTILIZATION_THRESHOLD:
            reasons.append(
                f"Expenditure utilization is only {r['utilization_pct']:.1f}% "
                f"of funds received, below the {LOW_UTILIZATION_THRESHOLD:.0f}% warning threshold"
            )
            points += 2

        if r["unreleased_share_pct"] >= UNRELEASED_SHARE_THRESHOLD:
            reasons.append(
                f"{r['unreleased_share_pct']:.1f}% of reported entitlement remains unreleased"
            )
            points += 1

        peer = state_avg.get(r["state"], national_util)
        if r["utilization_pct"] + 20 < peer:
            reasons.append(
                f"Utilization is materially below the state peer average ({peer:.1f}%)"
            )
            points += 1

        if bool(r["ml_anomaly"]):
            reasons.append(
                "ML cross-check detected an unusual combination of financial metrics"
            )
            points += 1

        if points >= 3:
            level = "High"
        elif points == 2:
            level = "Medium"
        elif points == 1:
            level = "Low"
        else:
            level = "None"

        report.append({
            "mp_id": r["mp_id"],
            "mp_name": r["mp_name"],
            "state": r["state"],
            "constituency": r["constituency"],
            "district": r["district"],
            "house": r["house"],
            "risk_level": level,
            "risk_points": int(points),
            "reasons": reasons,
            "ml_anomaly": bool(r["ml_anomaly"]),
            "own_utilization_pct": round(r["utilization_pct"], 1),
            "state_avg_utilization_pct": state_avg.get(r["state"], national_util),
            "national_avg_utilization_pct": national_util,
            "fund_received_cr": round(r["fund_received_cr"], 2),
            "recommended_cost_cr": round(r["recommended_cost_cr"], 2),
            "expenditure_cr": round(r["expenditure_cr"], 2),
            "entitlement_cr": round(r["entitlement_cr"], 2),
            "goi_release_cr": round(r["goi_release_cr"], 2),
            "unreleased_cr": round(r["unreleased_cr"], 2),
            "forecast": None,
            "data_availability": {
                "financial": True,
                "vendor": False,
                "work_status": False,
                "duplicate_work_text": False,
                "sc_st_compliance": False,
            },
        })

    return report, agg, national_util


def build_summaries(works, agg):
    state_summary = works.groupby("state").agg(
        sanctioned_cr=("sanctioned_amount_cr", "sum"),
        expenditure_cr=("final_expenditure_cr", "sum"),
        n_works=("work_id", "count"),
    ).reset_index()

    category_summary = works.groupby("category").agg(
        sanctioned_cr=("sanctioned_amount_cr", "sum")
    ).reset_index()

    # The supplied real file is an aggregate snapshot rather than a true
    # multi-year work-level time series, so do not invent a historical trend.
    yearly_trend = [{
        "year": "Source snapshot",
        "sanctioned_cr": round(works["sanctioned_amount_cr"].sum(), 1),
        "expenditure_cr": round(works["final_expenditure_cr"].sum(), 1),
    }]

    return state_summary, category_summary, yearly_trend


def main():
    works, roster = load_data()
    report, agg, national_util = build_report(works, roster)
    state_summary, category_summary, yearly_trend = build_summaries(works, agg)

    output = {
        "generated_at": datetime.now().isoformat(),
        "data_mode": "REAL",
        "data_notes": [
            "Core financial fields are from the supplied real MPLADS records.",
            "Vendor, completion-status, duplicate-work and SC/ST fields are not present in these source files.",
            "Forecasting is disabled because the supplied files are not a true multi-year work-level time series."
        ],
        "summary": {
            "total_mps": len(report),
            "total_works": int(len(works)),
            "completed_works": None,
            "total_sanctioned_cr": round(float(works["sanctioned_amount_cr"].sum()), 1),
            "total_expenditure_cr": round(float(works["final_expenditure_cr"].sum()), 1),
            "high_risk_count": sum(x["risk_level"] == "High" for x in report),
            "medium_risk_count": sum(x["risk_level"] == "Medium" for x in report),
            "low_risk_count": sum(x["risk_level"] == "Low" for x in report),
            "national_avg_utilization_pct": national_util,
        },
        "mps": report,
        "state_summary": state_summary.to_dict(orient="records"),
        "category_summary": category_summary.to_dict(orient="records"),
        "yearly_trend": yearly_trend,
        "state_yearly_trend": {},
        "state_category_summary": {},
        "mp_yearly_trend": {},
    }

    out = DATA_DIR / "mp_risk_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Wrote {out}")
    print(output["summary"])


if __name__ == "__main__":
    main()
