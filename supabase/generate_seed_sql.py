"""
MPLAD Drishti — Phase 1 seed SQL generator
============================================
Reads the REAL mp_roster.csv already in this repo (derived from the
RS_Session Rajya Sabha datasets) and generates INSERT statements for
states, districts, and mps. No invented rows — every state, district,
and MP here is one that already exists in your real data.

Run: python supabase/generate_seed_sql.py
Output: supabase/03_seed_data.sql

Then run 03_seed_data.sql THIRD in the Supabase SQL Editor, after
01_schema.sql and 02_rls_policies.sql have both succeeded.
"""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
roster = pd.read_csv(ROOT / "data" / "mp_roster.csv")

def esc(val):
    """SQL-escape a string value, or return NULL for missing data."""
    if pd.isna(val) or val is None or str(val).strip() == "":
        return "NULL"
    return "'" + str(val).replace("'", "''") + "'"

lines = []
lines.append("-- Auto-generated from data/mp_roster.csv — do not hand-edit.")
lines.append("-- Regenerate with: python supabase/generate_seed_sql.py\n")

# ---------- states ----------
states = sorted(roster["state"].dropna().unique())
lines.append("-- ---------- states ----------")
lines.append("insert into states (name) values")
lines.append(",\n".join(f"  ({esc(s)})" for s in states))
lines.append("on conflict (name) do nothing;\n")

# ---------- districts ----------
# Only real (state, district) pairs that actually appear in the data.
district_pairs = (
    roster.dropna(subset=["district"])[["state", "district"]]
    .drop_duplicates()
    .sort_values(["state", "district"])
)
lines.append("-- ---------- districts ----------")
lines.append("insert into districts (state_id, name)")
lines.append("select s.id, d.district_name from states s join (values")
district_rows = [
    f"  ({esc(r.state)}, {esc(r.district)})" for r in district_pairs.itertuples()
]
lines.append(",\n".join(district_rows))
lines.append(") as d(state_name, district_name) on s.name = d.state_name")
lines.append("on conflict (state_id, name) do nothing;\n")

# ---------- mps ----------
lines.append("-- ---------- mps ----------")
lines.append(
    "insert into mps (legacy_mp_id, mp_name, state_id, district_id, constituency, "
    "house, entitlement_cr, goi_release_cr, unreleased_cr, tenure, nodal_district, data_source)"
)
lines.append("select")
lines.append("  m.legacy_mp_id, m.mp_name, s.id, d.id, m.constituency, m.house,")
lines.append("  m.entitlement_cr, m.goi_release_cr, m.unreleased_cr, m.tenure,")
lines.append("  m.nodal_district, m.data_source")
lines.append("from (values")

mp_rows = []
for r in roster.itertuples():
    mp_rows.append(
        "  (" + ", ".join([
            esc(r.mp_id),
            esc(r.mp_name),
            esc(r.state),
            esc(r.district),  # used only to join to the right district row below
            esc(r.constituency),
            esc(r.house),
            "NULL" if pd.isna(r.entitlement_cr) else str(r.entitlement_cr),
            "NULL" if pd.isna(r.goi_release_cr) else str(r.goi_release_cr),
            "NULL" if pd.isna(r.unreleased_cr) else str(r.unreleased_cr),
            esc(r.tenure),
            esc(r.nodal_district),
            esc(r.data_source),
        ]) + ")"
    )
lines.append(",\n".join(mp_rows))
lines.append(
    ") as m(legacy_mp_id, mp_name, state_name, district_name, constituency, house, "
    "entitlement_cr, goi_release_cr, unreleased_cr, tenure, nodal_district, data_source)"
)
lines.append("join states s on s.name = m.state_name")
lines.append("left join districts d on d.state_id = s.id and d.name = m.district_name")
lines.append("on conflict (legacy_mp_id) do nothing;\n")

out_path = ROOT / "supabase" / "03_seed_data.sql"
out_path.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {out_path}")
print(f"States: {len(states)}, District pairs: {len(district_pairs)}, MPs: {len(roster)}")