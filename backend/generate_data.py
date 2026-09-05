"""
MPLAD Drishti - Real Data Normalizer
====================================
Reads the real MPLADS CSV files stored in ../data/real/ and converts them
into the stable files consumed by detect_anomalies.py and dashboard.html.

Real data available:
1) RS_Session_259_AU_2235_1.csv
   MP + district + fund received + recommended work cost + actual expenditure
2) RS_Session_256_AU_2872_2.csv
   MP + state + nodal district + tenure + entitlement + GOI release + unreleased

Important:
- This version does NOT invent vendors, work categories, completion dates,
  or SC/ST flags.
- Features that require those fields are marked unavailable by the detector.
- Financial risk analysis is calculated from the real records.
"""

from pathlib import Path
import re
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
REAL_DIR = ROOT_DIR / "data" / "real"
DATA_DIR = ROOT_DIR / "data"

FILE_1 = REAL_DIR / "RS_Session_259_AU_2235_1.csv"
FILE_2 = REAL_DIR / "RS_Session_256_AU_2872_2.csv"

# The raw government CSVs include trailing summary rows ("Total", "Grand
# Total") that are NOT real MPs. Applied consistently everywhere a row
# from the raw source is turned into an MP or a work/financial record —
# missing this in even one place lets a junk row leak back in via a
# different code path (confirmed: it did, via build_financial_records).
SUMMARY_ROW_NAMES = {"total", "grand total", "sub total", "subtotal"}


def is_summary_row(name):
    return str(name).strip().lower() in SUMMARY_ROW_NAMES


def clean_number(series):
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.replace("₹", "", regex=False).str.strip(),
        errors="coerce"
    ).fillna(0.0)


def clean_name(value):
    value = str(value).strip()
    value = re.sub(r"^(Shri|Smt\.?|Dr\.?|Ms\.?|Mr\.?|Prof\.?|Adv\.?)\s+", "", value, flags=re.I)
    return re.sub(r"\s+", " ", value).strip()


def norm_key(value):
    return re.sub(r"[^a-z0-9]", "", clean_name(value).lower())


def load_real_files():
    if not FILE_1.exists():
        raise FileNotFoundError(f"Missing real data file: {FILE_1}")
    if not FILE_2.exists():
        raise FileNotFoundError(f"Missing real data file: {FILE_2}")

    a = pd.read_csv(FILE_1)
    b = pd.read_csv(FILE_2)

    a["mp_key"] = a["MP Name"].map(norm_key)
    b["mp_key"] = b["Name of MP (Shri/Smt/Dr./Ms/Prof./Adv.)"].map(norm_key)
    return a, b


def build_roster(a, b):
    rows = []

    # Dataset 1 is the primary source for state + district + MP.
    # See SUMMARY_ROW_NAMES above for why summary rows are skipped here.
    for _, r in a.iterrows():
        name = clean_name(r["MP Name"])
        if is_summary_row(name):
            continue
        rows.append({
            "mp_id": "",
            "mp_name": name,
            "state": str(r["State/UT"]).strip(),
            "constituency": str(r["District"]).strip(),
            "district": str(r["District"]).strip(),
            "house": "Parliamentary Member",
            "data_source": "RS_Session_259_AU_2235_1.csv",
        })

    roster = pd.DataFrame(rows).drop_duplicates(subset=["mp_key"] if "mp_key" in rows[0] else ["mp_name"])

    # Assign stable IDs.
    roster = roster.reset_index(drop=True)
    roster["mp_id"] = [f"MP{n:04d}" for n in range(1, len(roster) + 1)]

    # Add release/entitlement information from dataset 2.
    release = b[[
        "mp_key",
        "Entitlement till his tenure",
        "GOI Release (in Cr)",
        "Unreleased Amount (in Cr)",
        "Tenure of MP",
        "Nodal District",
    ]].copy()
    release.columns = [
        "mp_key", "entitlement_cr", "goi_release_cr",
        "unreleased_cr", "tenure", "nodal_district"
    ]

    roster["mp_key"] = roster["mp_name"].map(norm_key)
    roster = roster.merge(release, on="mp_key", how="left")

    # Prefer nodal district where available, otherwise use dataset-1 district.
    roster["district"] = roster["nodal_district"].fillna(roster["district"])
    roster["entitlement_cr"] = roster["entitlement_cr"].fillna(0)
    roster["goi_release_cr"] = roster["goi_release_cr"].fillna(0)
    roster["unreleased_cr"] = roster["unreleased_cr"].fillna(0)
    roster["tenure"] = roster["tenure"].fillna("")

    return roster


def build_financial_records(a, roster):
    # Dataset 1 contains one financial aggregate per MP/district record.
    a = a.copy()
    a = a[~a["MP Name"].map(is_summary_row)]  # drop the same junk rows here too
    a["mp_key"] = a["MP Name"].map(norm_key)
    a["fund_received_cr"] = clean_number(a["Fund Received Goi (Rs. Crore)"])
    a["recommended_cost_cr"] = clean_number(a["Works Recommended Cost (Rs. Crore)"])
    a["actual_expenditure_cr"] = clean_number(
        a["Actual Expenditure Incurred with Exp_Admin (Rs. Crore)"]
    )

    a = a.merge(roster[["mp_id", "mp_key"]], on="mp_key", how="left")

    # One normalized "financial record" per real source row.
    records = pd.DataFrame({
        "work_id": ["REAL-" + str(x) for x in a["Sl. No."]],
        "mp_id": a["mp_id"],
        "mp_name": a["MP Name"].map(clean_name),
        "state": a["State/UT"].astype(str).str.strip(),
        "constituency": a["District"].astype(str).str.strip(),
        "district": a["District"].astype(str).str.strip(),
        "house": "Parliamentary Member",
        "year": 2025,
        "category": "All recommended works",
        "fund_received_cr": a["fund_received_cr"],
        "sanctioned_amount_cr": a["recommended_cost_cr"],
        "final_expenditure_cr": a["actual_expenditure_cr"],
        "sanction_date": pd.NaT,
        "status": "Unknown",
        "completion_date": pd.NaT,
        "implementing_agency": "Not available in source",
        "sc_st_area": pd.NA,
        "data_source": "RS_Session_259_AU_2235_1.csv",
    })

    # Useful derived metric.
    records["utilization_pct"] = (
        records["final_expenditure_cr"] / records["fund_received_cr"].replace(0, pd.NA) * 100
    ).fillna(0).round(1)

    return records


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    a, b = load_real_files()

    roster = build_roster(a, b)
    records = build_financial_records(a, roster)

    roster_cols = [
        "mp_id", "mp_name", "state", "constituency", "district", "house",
        "entitlement_cr", "goi_release_cr", "unreleased_cr", "tenure", "nodal_district",
        "data_source"
    ]
    roster[roster_cols].to_csv(DATA_DIR / "mp_roster.csv", index=False)
    records.to_csv(DATA_DIR / "mplads_works.csv", index=False)

    print(f"Real records loaded: {len(records)}")
    print(f"Unique MPs: {roster['mp_id'].nunique()}")
    print(f"Saved: {DATA_DIR / 'mp_roster.csv'}")
    print(f"Saved: {DATA_DIR / 'mplads_works.csv'}")


if __name__ == "__main__":
    main()