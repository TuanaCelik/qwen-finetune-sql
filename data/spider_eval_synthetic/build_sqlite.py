#!/usr/bin/env python3
"""
Generate synthetic department / management / head CSVs (~110 table rows) and
import them into SQLite for ad-hoc SQL checks (same DDL shape as sql_compare_ui/prompting.py).

Usage (from repo root):
  uv run python data/spider_eval_synthetic/build_sqlite.py
  uv run python data/spider_eval_synthetic/build_sqlite.py --out-dir /tmp/synth

Upload to GCS (example):
  gcloud storage cp /tmp/synth/*.csv gs://YOUR_BUCKET/path/synth/
  gcloud storage cp /tmp/synth/synthetic.db gs://YOUR_BUCKET/path/synth/
"""
from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path


def _departments() -> list[tuple[str, str, str]]:
    names = [
        ("D01", "Treasury", "1789"),
        ("D02", "Homeland Security", "2002"),
        ("D03", "Commerce", "1903"),
        ("D04", "Defense", "1947"),
        ("D05", "Justice", "1870"),
        ("D06", "Interior", "1849"),
        ("D07", "Agriculture", "1889"),
        ("D08", "Labor", "1913"),
        ("D09", "Energy", "1977"),
        ("D10", "Education", "1979"),
        ("D11", "HUD", "1965"),
        ("D12", "Transportation", "1966"),
        ("D13", "HHS", "1953"),
        ("D14", "VA", "1989"),
        ("D15", "State", "1789"),
        ("D16", "OMB", "1970"),
        ("D17", "EPA", "1970"),
        ("D18", "SBA", "1953"),
        ("D19", "Treasury Inspector General", "1989"),
        ("D20", "Federal Register", "1935"),
        ("D21", "GSA", "1949"),
        ("D22", "NASA", "1958"),
    ]
    return names


def _heads() -> list[tuple[str, str, str]]:
    # H01/H02 both Alabama — supports INTERSECT (Treasury vs Homeland) with different heads.
    rows: list[tuple[str, str, str]] = [
        ("H01", "Alice Smith", "Alabama"),
        ("H02", "Bob Jones", "Alabama"),
        ("H03", "Carol White", "Texas"),
        ("H04", "Dan Brown", "Texas"),
        ("H05", "Eve Davis", "California"),
        ("H06", "Frank Miller", "California"),
        ("H07", "Grace Lee", "New York"),
        ("H08", "Henry Wilson", "New York"),
        ("H09", "Ivy Chen", "Florida"),
        ("H10", "Jack Taylor", "Florida"),
        ("H11", "Karen Adams", "Ohio"),
        ("H12", "Leo Martinez", "Ohio"),
        ("H13", "Mia Thompson", "Georgia"),
        ("H14", "Noah Garcia", "Georgia"),
        ("H15", "Olivia Rodriguez", "Illinois"),
        ("H16", "Paul Nguyen", "Illinois"),
        ("H17", "Quinn Patel", "Pennsylvania"),
        ("H18", "Rita Okafor", "Pennsylvania"),
        ("H19", "Sam Okonkwo", "Michigan"),
        ("H20", "Tina Kowalski", "Michigan"),
        ("H21", "Uma Desai", "North Carolina"),
        ("H22", "Vik Singh", "North Carolina"),
        ("H23", "Wendy Clark", "Virginia"),
        ("H24", "Xavier Brooks", "Virginia"),
        ("H25", "Yuki Tanaka", "Washington"),
        ("H26", "Zara Khan", "Washington"),
        ("H27", "Aaron Berg", "Colorado"),
        ("H28", "Beth Cohen", "Colorado"),
        ("H29", "Chris Murphy", "Arizona"),
        ("H30", "Dana Ortiz", "Arizona"),
        ("H31", "Eric Lind", "Massachusetts"),
        ("H32", "Fiona Walsh", "Massachusetts"),
        ("H33", "Gina Park", "Oregon"),
        ("H34", "Hank Ruiz", "Oregon"),
        ("H35", "Iris Bloom", "Minnesota"),
        ("H36", "Jake Stone", "Minnesota"),
        ("H37", "Kelly Frost", "Wisconsin"),
        ("H38", "Liam Hart", "Wisconsin"),
        ("H39", "Mona Shah", "Tennessee"),
        ("H40", "Nick Vance", "Tennessee"),
    ]
    return rows


def _management() -> list[tuple[str, str, str]]:
    """(department_id, head_id, temporary_acting). D16–D18 have no rows (NOT IN demos). D01/D02 multi-row for HAVING."""
    rows: list[tuple[str, str, str]] = [
        ("D01", "H01", "No"),
        ("D01", "H03", "Yes"),
        ("D02", "H02", "No"),
        ("D02", "H04", "Yes"),
        ("D03", "H05", "No"),
        ("D04", "H06", "No"),
        ("D05", "H07", "Yes"),
        ("D06", "H08", "No"),
        ("D07", "H09", "No"),
        ("D08", "H10", "Yes"),
        ("D09", "H11", "No"),
        ("D10", "H12", "No"),
        ("D11", "H13", "Yes"),
        ("D12", "H14", "No"),
        ("D13", "H15", "No"),
        ("D14", "H16", "Yes"),
        ("D15", "H17", "No"),
        ("D01", "H18", "No"),
        ("D02", "H19", "No"),
        ("D03", "H20", "Yes"),
        ("D04", "H21", "No"),
        ("D05", "H22", "No"),
        ("D06", "H23", "Yes"),
        ("D07", "H24", "No"),
        ("D08", "H25", "No"),
        ("D09", "H26", "Yes"),
        ("D10", "H27", "No"),
        ("D11", "H28", "No"),
        ("D12", "H29", "Yes"),
        ("D13", "H30", "No"),
        ("D14", "H31", "No"),
        ("D15", "H32", "Yes"),
        ("D19", "H33", "No"),
        ("D20", "H34", "Yes"),
        ("D21", "H35", "No"),
        ("D22", "H36", "No"),
        ("D19", "H37", "Yes"),
        ("D20", "H38", "No"),
        ("D21", "H39", "Yes"),
        ("D22", "H40", "No"),
        ("D03", "H27", "No"),
        ("D04", "H28", "Yes"),
        ("D05", "H29", "No"),
        ("D06", "H30", "No"),
        ("D07", "H31", "Yes"),
        ("D08", "H32", "No"),
        ("D09", "H33", "No"),
        ("D10", "H34", "Yes"),
    ]
    return rows


def write_csv(path: Path, header: list[str], rows: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def build_sqlite(db_path: Path) -> None:
    dept = _departments()
    heads = _heads()
    mgmt = _management()

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.executescript(
        """
        PRAGMA foreign_keys = OFF;
        DROP TABLE IF EXISTS management;
        DROP TABLE IF EXISTS department;
        DROP TABLE IF EXISTS head;
        CREATE TABLE department (
          department_id TEXT PRIMARY KEY,
          name TEXT NOT NULL,
          creation TEXT NOT NULL
        );
        CREATE TABLE head (
          head_id TEXT PRIMARY KEY,
          name TEXT NOT NULL,
          born_state TEXT NOT NULL
        );
        CREATE TABLE management (
          department_id TEXT NOT NULL,
          head_id TEXT NOT NULL,
          temporary_acting TEXT NOT NULL,
          PRIMARY KEY (department_id, head_id)
        );
        """
    )
    cur.executemany(
        "INSERT INTO department (department_id, name, creation) VALUES (?,?,?)",
        dept,
    )
    cur.executemany(
        "INSERT INTO head (head_id, name, born_state) VALUES (?,?,?)",
        heads,
    )
    cur.executemany(
        "INSERT INTO management (department_id, head_id, temporary_acting) VALUES (?,?,?)",
        mgmt,
    )
    conn.commit()
    conn.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory for CSVs + synthetic.db (default: this folder)",
    )
    args = p.parse_args()
    out: Path = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    dept = _departments()
    heads = _heads()
    mgmt = _management()

    write_csv(out / "department.csv", ["department_id", "name", "creation"], dept)
    write_csv(out / "head.csv", ["head_id", "name", "born_state"], heads)
    write_csv(out / "management.csv", ["department_id", "head_id", "temporary_acting"], mgmt)

    db_path = out / "synthetic.db"
    build_sqlite(db_path)

    n = len(dept) + len(heads) + len(mgmt)
    print(f"Wrote {len(dept)} department, {len(heads)} head, {len(mgmt)} management rows ({n} total).")
    print(f"CSV dir: {out.resolve()}")
    print(f"SQLite:  {db_path.resolve()}")


if __name__ == "__main__":
    main()
