from pathlib import Path
import csv
import sqlite3

PROJECT_DIR = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_DIR / "university_performance.db"
QUERY_PATH = PROJECT_DIR / "sql" / "regional_outcome_summary.sql"
REPORT_PATH = PROJECT_DIR / "reports" / "regional_outcome_summary.csv"


def build_regional_summary():
    """Create a regional summary report from the practice database."""
    query = QUERY_PATH.read_text()                  

    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(query).fetchall()

    REPORT_PATH.parent.mkdir(exist_ok=True)
    with REPORT_PATH.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["region", "universities", "average_satisfaction", "strong_outcomes"])
        writer.writerows(rows)

    return rows


if __name__ == "__main__":
    summary_rows = build_regional_summary()
    print(f"Created {REPORT_PATH.name} with {len(summary_rows)} rows")
