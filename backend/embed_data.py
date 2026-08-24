"""
Re-embeds the latest data/mp_risk_report.json into frontend/dashboard.html
so the dashboard stays a single self-contained file (no server needed).

Run this after every detect_anomalies.py run if you've changed the
detection logic and want the dashboard to reflect it:

    python backend/generate_data.py
    python backend/detect_anomalies.py
    python backend/embed_data.py
"""
import json
import re
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "mp_risk_report.json")
DASHBOARD_PATH = os.path.join(PROJECT_ROOT, "frontend", "dashboard.html")

with open(DATA_PATH, encoding="utf-8") as f:
    data_json = json.dumps(json.load(f))

with open(DASHBOARD_PATH, encoding="utf-8") as f:
    html = f.read()

new_html, n = re.subn(
    r"const REPORT = \{.*?\}\s*;",
    lambda m: f"const REPORT = {data_json};",
    html,
    flags=re.DOTALL,
)

if n == 0:
    raise RuntimeError("Could not find 'const REPORT = {...};' in dashboard.html -- check the file wasn't edited in a way that broke the pattern.")

with open(DASHBOARD_PATH, "w", encoding="utf-8") as f:
    f.write(new_html)

print(f"Dashboard updated with latest data ({n} replacement made).")