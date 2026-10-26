"""Fill {{table:name}} placeholders in reports/report_template.md from reports/tables/*.csv -> final_report.md."""
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pdm import REPORTS


def md(name: str) -> str:
    df = pd.read_csv(REPORTS / "tables" / f"{name}.csv")
    df = df.round(3)
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    body = "\n".join("| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |" for r in df.itertuples(index=False))
    return f"{head}{body}\n\n*Source: `reports/tables/{name}.csv`*"


tpl = (REPORTS / "report_template.md").read_text()
out = re.sub(r"\{\{table:(\w+)\}\}", lambda m: md(m.group(1)), tpl)
(REPORTS / "final_report.md").write_text(out)
print("wrote final_report.md,", len(out.split()), "words")
