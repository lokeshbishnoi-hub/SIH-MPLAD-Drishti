"""
MPLAD Drishti - Synthetic Data Generator
==========================================
Generates a MPLADS works dataset matching the schema of the public
MoSPI/Dataful dataset, for use until we have full portal/API access.
Column names match the real dataset so detect_anomalies.py can be
pointed at live data later with no changes needed.

Real data references:
  - data.gov.in -> "Utilisation of MPLAD Scheme Funds"
  - dataful.in/datasets/18542 (MoSPI)
  - mplads.mospi.gov.in (eSAKSHI portal)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

STATES = [
    "Uttar Pradesh", "Maharashtra", "West Bengal", "Bihar", "Tamil Nadu",
    "Madhya Pradesh", "Karnataka", "Gujarat", "Rajasthan", "Andhra Pradesh",
    "Odisha", "Telangana", "Kerala", "Jharkhand", "Assam", "Punjab",
    "Chhattisgarh", "Haryana", "Delhi", "Uttarakhand"
]

# MPs per state roughly proportional to real Lok Sabha seat counts (simplified)
MPS_PER_STATE = {
    "Uttar Pradesh": 16, "Maharashtra": 10, "West Bengal": 8, "Bihar": 8,
    "Tamil Nadu": 8, "Madhya Pradesh": 6, "Karnataka": 6, "Gujarat": 6,
    "Rajasthan": 6, "Andhra Pradesh": 5, "Odisha": 5, "Telangana": 4,
    "Kerala": 5, "Jharkhand": 4, "Assam": 4, "Punjab": 4,
    "Chhattisgarh": 3, "Haryana": 3, "Delhi": 2, "Uttarakhand": 2,
}

CATEGORIES = [
    "Roads, Pathways & Bridges", "Drinking Water", "Education",
    "Health & Family Welfare", "Electricity Facility", "Irrigation",
    "Other Public Facilities (Community Centres)", "Sports Infrastructure",
    "Sanitation", "Rural Development",
]

# Real published national average: ~91% utilization (Deloitte/MoSPI eval report)
NATIONAL_AVG_UTILIZATION = 0.91
ANNUAL_ENTITLEMENT_CR = 5.0  # Rs 5 Crore per MP per year

FIRST_NAMES = ["Rajesh", "Sunita", "Amit", "Priya", "Vikram", "Anjali", "Suresh",
               "Kavita", "Manoj", "Deepa", "Ashok", "Meena", "Ravi", "Pooja",
               "Sanjay", "Neha", "Arun", "Divya", "Vinod", "Shalini"]
LAST_NAMES = ["Sharma", "Verma", "Reddy", "Patel", "Singh", "Iyer", "Gupta",
              "Nair", "Yadav", "Chauhan", "Rao", "Das", "Mishra", "Joshi",
              "Kumar", "Pillai"]

VENDORS = [f"{a} Infrastructure {b}" for a, b in [
    ("Shree", "Pvt Ltd"), ("National", "Constructions"), ("Bharat", "Builders"),
    ("Om", "Enterprises"), ("Vishwakarma", "Contractors"), ("Ganesh", "Projects"),
    ("Sai", "Developers"), ("Raj", "Infra"), ("Progressive", "Works"),
    ("United", "Engineering Co"), ("Modern", "Construction"), ("Krishna", "Associates"),
]]


def make_mp_roster():
    mps = []
    mp_id = 1
    for state, count in MPS_PER_STATE.items():
        for i in range(count):
            name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
            mps.append({
                "mp_id": f"MP{mp_id:04d}",
                "mp_name": name,
                "state": state,
                "constituency": f"{state} Constituency {i+1}",
                "house": "Lok Sabha" if random.random() > 0.15 else "Rajya Sabha",
            })
            mp_id += 1
    return pd.DataFrame(mps)


def generate_works(mp_roster, years=(2021, 2022, 2023, 2024, 2025)):
    """Generate individual sanctioned works per MP per year."""
    works = []
    work_id = 1

    # Most MPs behave normally; a small set is seeded with a specific
    # anomaly type so the detection engine has real signal to catch.
    n_mps = len(mp_roster)
    anomaly_types_cycle = ["low_utilization", "cost_overrun", "vendor_concentration", "delayed_works"]
    n_anomalous = max(len(anomaly_types_cycle) * 2, n_mps // 12)
    anomalous_ids = list(mp_roster.sample(n_anomalous, random_state=1)["mp_id"])
    # cycle through types deterministically so every anomaly type is
    # guaranteed to appear (pure random assignment could skip a type)
    anomaly_type_map = {
        mp_id: anomaly_types_cycle[i % len(anomaly_types_cycle)]
        for i, mp_id in enumerate(anomalous_ids)
    }

    for _, mp in mp_roster.iterrows():
        anomaly_type = anomaly_type_map.get(mp["mp_id"])

        # vendor concentration setup
        if anomaly_type == "vendor_concentration":
            preferred_vendor = random.choice(VENDORS)

        for year in years:
            entitlement = ANNUAL_ENTITLEMENT_CR
            if anomaly_type == "low_utilization":
                utilization_rate = np.random.uniform(0.15, 0.35)
            else:
                utilization_rate = np.clip(np.random.normal(NATIONAL_AVG_UTILIZATION, 0.07), 0.4, 1.0)

            n_works = np.random.randint(3, 9)
            year_budget = entitlement * utilization_rate

            for _ in range(n_works):
                category = random.choice(CATEGORIES)
                sanctioned_amt = round(year_budget / n_works * np.random.uniform(0.7, 1.3), 3)

                if anomaly_type == "cost_overrun" and random.random() < 0.6:
                    final_expenditure = round(sanctioned_amt * np.random.uniform(1.4, 2.2), 3)
                else:
                    final_expenditure = round(sanctioned_amt * np.random.uniform(0.85, 1.05), 3)

                sanction_date = datetime(year, random.randint(1, 12), random.randint(1, 28))

                if anomaly_type == "delayed_works" and random.random() < 0.65:
                    status = "In Progress"
                    completion_date = None
                elif sanction_date < datetime(2024, 8, 1):
                    status = "Completed"
                    completion_date = sanction_date + timedelta(days=np.random.randint(90, 330))
                else:
                    status = random.choices(["Completed", "In Progress"], weights=[0.7, 0.3])[0]
                    completion_date = (sanction_date + timedelta(days=np.random.randint(90, 300))
                                        if status == "Completed" else None)

                if anomaly_type == "vendor_concentration" and random.random() < 0.75:
                    vendor = preferred_vendor
                else:
                    vendor = random.choice(VENDORS)

                sc_st_area = random.random() < 0.25
                works.append({
                    "work_id": f"W{work_id:06d}",
                    "mp_id": mp["mp_id"],
                    "mp_name": mp["mp_name"],
                    "state": mp["state"],
                    "constituency": mp["constituency"],
                    "house": mp["house"],
                    "year": year,
                    "category": category,
                    "sanctioned_amount_cr": sanctioned_amt,
                    "final_expenditure_cr": final_expenditure,
                    "sanction_date": sanction_date.date().isoformat(),
                    "status": status,
                    "completion_date": completion_date.date().isoformat() if completion_date else None,
                    "implementing_agency": vendor,
                    "sc_st_area": sc_st_area,
                })
                work_id += 1

    return pd.DataFrame(works)


if __name__ == "__main__":
    import os
    # resolve paths relative to this script so it runs from any machine
    # expects: <project_root>/backend/generate_data.py, <project_root>/data/
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
    os.makedirs(DATA_DIR, exist_ok=True)

    roster = make_mp_roster()
    works = generate_works(roster)
    roster.to_csv(os.path.join(DATA_DIR, "mp_roster.csv"), index=False)
    works.to_csv(os.path.join(DATA_DIR, "mplads_works.csv"), index=False)
    print(f"Generated {len(roster)} MPs and {len(works)} works records.")
    print(f"Saved to: {os.path.abspath(DATA_DIR)}")
    print(works.head())