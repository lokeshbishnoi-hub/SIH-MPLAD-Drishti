"""
MPLAD Drishti - Anomaly Detection Engine
==========================================
Detects trends, anomalies, cost overruns, duplicate works, delayed
projects, deviations from norms, and compliance gaps (e.g. SC/ST
allocation rule) in MPLADS fund utilization data, and produces
risk-scored, plain-language alerts per MP.

Run: python detect_anomalies.py
Outputs: ../data/mp_risk_report.json (consumed by the dashboard)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
import json
from datetime import datetime, date
from difflib import SequenceMatcher

import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

# --- MPLADS Guideline thresholds (2023 Guidelines) ---
SC_ALLOC_MIN_PCT = 15.0   # min % of funds for SC-population areas
ST_ALLOC_MIN_PCT = 7.5    # min % of funds for ST-population areas
COMPLETION_WINDOW_DAYS = 365  # works should complete within 1 year of sanction
COST_OVERRUN_THRESHOLD = 1.30  # >30% over sanctioned = flag
LOW_UTILIZATION_THRESHOLD = 0.60  # national avg is ~91%; <60% is a flag
VENDOR_CONCENTRATION_THRESHOLD = 0.50  # one vendor >50% of an MP's works


def load_data():
    works = pd.read_csv(f"{DATA_DIR}/mplads_works.csv", parse_dates=["sanction_date"])
    roster = pd.read_csv(f"{DATA_DIR}/mp_roster.csv")
    return works, roster


def flag_cost_overruns(works):
    works = works.copy()
    works["overrun_ratio"] = works["final_expenditure_cr"] / works["sanctioned_amount_cr"]
    flagged = works[works["overrun_ratio"] >= COST_OVERRUN_THRESHOLD]
    return flagged.groupby("mp_id").agg(
        overrun_work_count=("work_id", "count"),
        avg_overrun_pct=("overrun_ratio", lambda x: round((x.mean() - 1) * 100, 1)),
        total_excess_cr=("final_expenditure_cr", lambda x: round(
            (works.loc[x.index, "final_expenditure_cr"] - works.loc[x.index, "sanctioned_amount_cr"]).sum(), 2))
    ).reset_index()


def flag_low_utilization(works, roster, years=5):
    total_by_mp = works.groupby("mp_id")["sanctioned_amount_cr"].sum().reset_index()
    total_by_mp["entitlement_cr"] = 5.0 * years
    total_by_mp["utilization_rate"] = total_by_mp["sanctioned_amount_cr"] / total_by_mp["entitlement_cr"]
    flagged = total_by_mp[total_by_mp["utilization_rate"] < LOW_UTILIZATION_THRESHOLD]
    flagged["utilization_pct"] = (flagged["utilization_rate"] * 100).round(1)
    return flagged[["mp_id", "utilization_pct"]]


def flag_vendor_concentration(works):
    results = []
    for mp_id, grp in works.groupby("mp_id"):
        vendor_counts = grp["implementing_agency"].value_counts(normalize=True)
        top_vendor = vendor_counts.index[0]
        top_share = vendor_counts.iloc[0]
        if top_share >= VENDOR_CONCENTRATION_THRESHOLD and len(grp) >= 5:
            national_avg_share = 1 / grp["implementing_agency"].nunique() if grp["implementing_agency"].nunique() else 1
            results.append({
                "mp_id": mp_id,
                "top_vendor": top_vendor,
                "top_vendor_share_pct": round(top_share * 100, 1),
                "concentration_multiple": round(top_share / max(1 / max(vendor_counts.shape[0], 1), 0.01), 1),
            })
    return pd.DataFrame(results)


def flag_delayed_works(works, as_of=None):
    """Flag MPs where an unusually HIGH SHARE of their older works remain
    incomplete -- not just MPs with any overdue work at all, since a
    couple of legitimately slow projects is normal and shouldn't itself
    be a red flag. We compare each MP's overdue rate against the
    peer average."""
    as_of = as_of or datetime(2026, 8, 24)
    works = works.copy()
    works["days_since_sanction"] = (as_of - works["sanction_date"]).dt.days
    eligible = works[works["days_since_sanction"] > COMPLETION_WINDOW_DAYS].copy()
    if eligible.empty:
        return pd.DataFrame(columns=["mp_id", "overdue_work_count", "overdue_rate_pct", "max_overdue_days"])

    eligible["is_overdue"] = eligible["status"] == "In Progress"
    per_mp = eligible.groupby("mp_id").agg(
        eligible_count=("work_id", "count"),
        overdue_work_count=("is_overdue", "sum"),
        max_overdue_days=("days_since_sanction", "max"),
    ).reset_index()
    per_mp["overdue_rate_pct"] = (per_mp["overdue_work_count"] / per_mp["eligible_count"] * 100).round(1)

    peer_avg_rate = per_mp["overdue_rate_pct"].mean()
    # flag only MPs meaningfully above peer average AND with a
    # non-trivial number of overdue works
    flagged = per_mp[
        (per_mp["overdue_rate_pct"] > max(peer_avg_rate * 1.8, 40)) &
        (per_mp["overdue_work_count"] >= 3)
    ]
    return flagged[["mp_id", "overdue_work_count", "overdue_rate_pct", "max_overdue_days"]]


def flag_duplicate_works(works, similarity_threshold=0.85):
    """Flag works with near-identical category+agency+amount combos for the same MP/year --
    a lightweight proxy for 'duplicate works', explainable without heavy NLP."""
    dup_flags = []
    for (mp_id, year), grp in works.groupby(["mp_id", "year"]):
        seen = []
        for _, row in grp.iterrows():
            sig = (row["category"], row["implementing_agency"], round(row["sanctioned_amount_cr"], 1))
            if sig in seen:
                dup_flags.append(mp_id)
            seen.append(sig)
    if not dup_flags:
        return pd.DataFrame(columns=["mp_id", "duplicate_count"])
    s = pd.Series(dup_flags).value_counts().reset_index()
    s.columns = ["mp_id", "duplicate_count"]
    return s


def flag_sc_st_noncompliance(works):
    results = []
    for mp_id, grp in works.groupby("mp_id"):
        total = grp["sanctioned_amount_cr"].sum()
        sc_st_amt = grp.loc[grp["sc_st_area"], "sanctioned_amount_cr"].sum()
        pct = (sc_st_amt / total * 100) if total > 0 else 0
        # Combined 15%+7.5% threshold used as a single simplified check;
        # production version should track SC and ST populations separately.
        if pct < (SC_ALLOC_MIN_PCT):
            results.append({"mp_id": mp_id, "sc_st_alloc_pct": round(pct, 1)})
    return pd.DataFrame(results)


def isolation_forest_flags(works):
    """ML-based anomaly detection as a cross-check on the rule-based
    flags -- catches unusual combinations the explicit rules might miss."""
    agg = works.groupby("mp_id").agg(
        total_sanctioned=("sanctioned_amount_cr", "sum"),
        total_expenditure=("final_expenditure_cr", "sum"),
        n_works=("work_id", "count"),
        n_vendors=("implementing_agency", "nunique"),
        avg_overrun=("final_expenditure_cr", lambda x: (x / works.loc[x.index, "sanctioned_amount_cr"]).mean()),
        pct_in_progress=("status", lambda x: (x == "In Progress").mean()),
    ).reset_index()

    features = agg[["total_sanctioned", "total_expenditure", "n_works",
                     "n_vendors", "avg_overrun", "pct_in_progress"]].fillna(0)

    iso = IsolationForest(contamination=0.12, random_state=42)
    agg["ml_anomaly"] = iso.fit_predict(features) == -1
    agg["ml_anomaly_score"] = -iso.decision_function(features)  # higher = more anomalous
    return agg[["mp_id", "ml_anomaly", "ml_anomaly_score"]]


def forecast_year_end_utilization(works, current_year=2026):
    """Simple linear trend forecast: given each MP's utilization pattern
    over prior years, project where they'll land by year-end if the
    trend continues. This is the 'predictive insights' capability the
    problem statement calls for -- kept intentionally simple (linear
    regression on yearly utilization) so it stays explainable rather
    than a black box."""
    yearly = works.groupby(["mp_id", "year"])["sanctioned_amount_cr"].sum().reset_index()
    forecasts = []

    for mp_id, grp in yearly.groupby("mp_id"):
        grp = grp.sort_values("year")
        if len(grp) < 3:
            continue  # not enough history for a meaningful trend
        X = grp["year"].values.reshape(-1, 1)
        y = (grp["sanctioned_amount_cr"].values / 5.0) * 100  # as % of annual entitlement

        model = LinearRegression()
        model.fit(X, y)
        projected_pct = float(model.predict([[current_year]])[0])
        projected_pct = round(max(0, min(150, projected_pct)), 1)

        recent_avg = round(y[-2:].mean(), 1)
        trend_direction = "declining" if model.coef_[0] < -1 else ("improving" if model.coef_[0] > 1 else "stable")

        forecasts.append({
            "mp_id": mp_id,
            "projected_year_end_utilization_pct": projected_pct,
            "recent_avg_utilization_pct": recent_avg,
            "trend": trend_direction,
        })

    return pd.DataFrame(forecasts)


def build_risk_report():
    works, roster = load_data()

    cost_overruns = flag_cost_overruns(works)
    low_util = flag_low_utilization(works, roster)
    vendor_conc = flag_vendor_concentration(works)
    delayed = flag_delayed_works(works)
    duplicates = flag_duplicate_works(works)
    sc_st = flag_sc_st_noncompliance(works)
    ml_flags = isolation_forest_flags(works)
    forecasts = forecast_year_end_utilization(works)

    # peer benchmarks used for the comparison toggle in the dashboard
    national_avg_utilization = round(
        (works.groupby("mp_id")["sanctioned_amount_cr"].sum() / 25.0 * 100).mean(), 1
    )  # 25 = 5 Cr/year x 5 years
    state_avg_utilization = (
        works.groupby(["state", "mp_id"])["sanctioned_amount_cr"].sum()
        .reset_index().groupby("state")["sanctioned_amount_cr"]
        .mean().apply(lambda x: round(x / 25.0 * 100, 1)).to_dict()
    )

    report = {}
    for _, mp in roster.iterrows():
        mp_id = mp["mp_id"]
        reasons = []
        risk_points = 0

        row = cost_overruns[cost_overruns["mp_id"] == mp_id]
        if not row.empty:
            r = row.iloc[0]
            reasons.append(f"{int(r['overrun_work_count'])} works exceeded sanctioned cost by an average of {r['avg_overrun_pct']}% (total excess: ₹{r['total_excess_cr']} Cr)")
            risk_points += 2

        row = low_util[low_util["mp_id"] == mp_id]
        if not row.empty:
            r = row.iloc[0]
            reasons.append(f"Fund utilization is only {r['utilization_pct']}%, well below the national average of ~91%")
            risk_points += 2

        row = vendor_conc[vendor_conc["mp_id"] == mp_id] if not vendor_conc.empty else pd.DataFrame()
        if not row.empty:
            r = row.iloc[0]
            reasons.append(f"{r['top_vendor_share_pct']}% of works awarded to a single implementing agency ({r['top_vendor']})")
            risk_points += 2

        row = delayed[delayed["mp_id"] == mp_id]
        if not row.empty:
            r = row.iloc[0]
            reasons.append(f"{int(r['overdue_work_count'])} works ({r['overdue_rate_pct']}% of eligible works) remain incomplete beyond the 1-year guideline, up to {int(r['max_overdue_days'])} days overdue -- well above the peer average")
            risk_points += 2

        row = duplicates[duplicates["mp_id"] == mp_id] if not duplicates.empty else pd.DataFrame()
        if not row.empty:
            r = row.iloc[0]
            reasons.append(f"{int(r['duplicate_count'])} potentially duplicate work(s) detected (same category, agency & cost within a year)")
            risk_points += 1

        row = sc_st[sc_st["mp_id"] == mp_id] if not sc_st.empty else pd.DataFrame()
        if not row.empty:
            r = row.iloc[0]
            reasons.append(f"Only {r['sc_st_alloc_pct']}% of funds allocated to SC/ST areas (guideline minimum: 15%/7.5%)")
            risk_points += 1

        row = ml_flags[ml_flags["mp_id"] == mp_id]
        ml_anomalous = bool(row.iloc[0]["ml_anomaly"]) if not row.empty else False
        if ml_anomalous and risk_points == 0:
            reasons.append("Flagged by the ML anomaly model based on an unusual combination of spending metrics")
            risk_points += 1

        if risk_points >= 3:
            risk_level = "High"
        elif risk_points == 2:
            risk_level = "Medium"
        elif risk_points >= 1:
            risk_level = "Low"
        else:
            risk_level = "None"

        fc_row = forecasts[forecasts["mp_id"] == mp_id]
        forecast_data = None
        if not fc_row.empty:
            fr = fc_row.iloc[0]
            forecast_data = {
                "projected_year_end_utilization_pct": fr["projected_year_end_utilization_pct"],
                "trend": fr["trend"],
            }

        mp_own_utilization = round(
            works[works["mp_id"] == mp_id]["sanctioned_amount_cr"].sum() / 25.0 * 100, 1
        )

        report[mp_id] = {
            "mp_id": mp_id,
            "mp_name": mp["mp_name"],
            "state": mp["state"],
            "constituency": mp["constituency"],
            "house": mp["house"],
            "risk_level": risk_level,
            "risk_points": risk_points,
            "reasons": reasons,
            "ml_anomaly": ml_anomalous,
            "forecast": forecast_data,
            "own_utilization_pct": mp_own_utilization,
            "state_avg_utilization_pct": state_avg_utilization.get(mp["state"]),
            "national_avg_utilization_pct": national_avg_utilization,
        }

    # National / state summary stats for the dashboard overview
    total_sanctioned = float(works["sanctioned_amount_cr"].sum())
    total_expenditure = float(works["final_expenditure_cr"].sum())
    total_works = int(len(works))
    completed_works = int((works["status"] == "Completed").sum())

    state_summary = works.groupby("state").agg(
        sanctioned_cr=("sanctioned_amount_cr", "sum"),
        expenditure_cr=("final_expenditure_cr", "sum"),
        n_works=("work_id", "count"),
    ).reset_index()
    state_summary["sanctioned_cr"] = state_summary["sanctioned_cr"].round(1)
    state_summary["expenditure_cr"] = state_summary["expenditure_cr"].round(1)

    category_summary = works.groupby("category").agg(
        sanctioned_cr=("sanctioned_amount_cr", "sum"),
    ).reset_index().sort_values("sanctioned_cr", ascending=False)
    category_summary["sanctioned_cr"] = category_summary["sanctioned_cr"].round(1)

    yearly_trend = works.groupby("year").agg(
        sanctioned_cr=("sanctioned_amount_cr", "sum"),
        expenditure_cr=("final_expenditure_cr", "sum"),
    ).reset_index()
    yearly_trend["sanctioned_cr"] = yearly_trend["sanctioned_cr"].round(1)
    yearly_trend["expenditure_cr"] = yearly_trend["expenditure_cr"].round(1)

    total_excess_cr = round(
        sum(r["total_excess_cr"] for r in cost_overruns.to_dict(orient="records")), 2
    ) if not cost_overruns.empty else 0.0

    output = {
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "total_mps": len(roster),
            "total_works": total_works,
            "completed_works": completed_works,
            "total_sanctioned_cr": round(total_sanctioned, 1),
            "total_expenditure_cr": round(total_expenditure, 1),
            "high_risk_count": sum(1 for v in report.values() if v["risk_level"] == "High"),
            "medium_risk_count": sum(1 for v in report.values() if v["risk_level"] == "Medium"),
            "low_risk_count": sum(1 for v in report.values() if v["risk_level"] == "Low"),
            "total_excess_cr": total_excess_cr,
            "national_avg_utilization_pct": national_avg_utilization,
        },
        "mps": list(report.values()),
        "state_summary": state_summary.to_dict(orient="records"),
        "category_summary": category_summary.to_dict(orient="records"),
        "yearly_trend": yearly_trend.to_dict(orient="records"),
    }
    return output


if __name__ == "__main__":
    report = build_risk_report()
    out_path = f"{DATA_DIR}/mp_risk_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote risk report to {out_path}")
    print(f"Summary: {report['summary']}")
    high_risk = [m for m in report["mps"] if m["risk_level"] == "High"]
    print(f"\n{len(high_risk)} High-risk MPs flagged, e.g.:")
    for m in high_risk[:3]:
        print(f"  - {m['mp_name']} ({m['constituency']}): {m['reasons']}")