"""
exporter.py
Export and import attendance / persons data as CSV or Excel (.xlsx).
"""

import os
import csv
import json
from datetime import datetime, date
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

EXPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "exports")
os.makedirs(EXPORT_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _try_pandas():
    try:
        import pandas as pd
        return pd
    except ImportError:
        return None


# ── Attendance Export ─────────────────────────────────────────────────────────

def export_attendance_csv(records: List[Dict],
                          filename: Optional[str] = None) -> str:
    """Write attendance records to CSV. Returns file path."""
    filename = filename or f"attendance_{_timestamp()}.csv"
    path = os.path.join(EXPORT_DIR, filename)

    fieldnames = ["date", "name", "employee_id", "department",
                  "check_in", "check_out", "status", "confidence"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            r["confidence"] = f"{r.get('confidence', 0):.2%}"
            writer.writerow(r)

    logger.info(f"Exported {len(records)} records → {path}")
    return path


def export_attendance_excel(records: List[Dict],
                            filename: Optional[str] = None) -> str:
    """Write attendance records to Excel (.xlsx). Returns file path."""
    pd = _try_pandas()
    if pd is None:
        raise ImportError("pandas is required for Excel export. Run: pip install pandas openpyxl")

    filename = filename or f"attendance_{_timestamp()}.xlsx"
    path = os.path.join(EXPORT_DIR, filename)

    cols = ["date", "name", "employee_id", "department",
            "check_in", "check_out", "status", "confidence"]
    df = pd.DataFrame(records)

    # Keep only existing columns in order
    present_cols = [c for c in cols if c in df.columns]
    df = df[present_cols]
    if "confidence" in df.columns:
        df["confidence"] = df["confidence"].apply(lambda x: f"{float(x):.2%}" if x else "")

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Attendance", index=False)

        # Auto-width columns
        ws = writer.sheets["Attendance"]
        for col in ws.columns:
            max_len = max((len(str(cell.value)) for cell in col if cell.value), default=10)
            ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 40)

    logger.info(f"Exported {len(records)} records → {path}")
    return path


def export_persons_csv(persons: List[Dict],
                       filename: Optional[str] = None) -> str:
    """Export registered persons (no embeddings) to CSV."""
    filename = filename or f"persons_{_timestamp()}.csv"
    path = os.path.join(EXPORT_DIR, filename)

    fieldnames = ["id", "name", "employee_id", "department",
                  "email", "phone", "registered_at", "is_active"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(persons)

    logger.info(f"Exported {len(persons)} persons → {path}")
    return path


# ── Attendance Import ─────────────────────────────────────────────────────────

def import_attendance_csv(filepath: str) -> List[Dict]:
    """
    Read attendance CSV and return list of dicts.
    Expected columns: date, employee_id, check_in, [check_out], [status]
    """
    records = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(dict(row))
    logger.info(f"Imported {len(records)} records from {filepath}")
    return records


def import_attendance_excel(filepath: str,
                            sheet: str = "Attendance") -> List[Dict]:
    """Read attendance from Excel."""
    pd = _try_pandas()
    if pd is None:
        raise ImportError("pandas is required for Excel import.")

    df = pd.read_excel(filepath, sheet_name=sheet, dtype=str)
    df = df.fillna("")
    records = df.to_dict(orient="records")
    logger.info(f"Imported {len(records)} records from {filepath}")
    return records


def import_persons_csv(filepath: str) -> List[Dict]:
    """Read persons template CSV."""
    records = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(dict(row))
    return records


def generate_persons_template(filename: Optional[str] = None) -> str:
    """Write an empty persons template CSV for bulk import."""
    filename = filename or "persons_template.csv"
    path = os.path.join(EXPORT_DIR, filename)
    fieldnames = ["name", "employee_id", "department", "email", "phone"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # Example row
        writer.writerow({
            "name": "John Doe",
            "employee_id": "EMP001",
            "department": "Engineering",
            "email": "john@example.com",
            "phone": "+91-9000000000"
        })
    return path


def generate_summary_excel(records: List[Dict],
                           filename: Optional[str] = None) -> str:
    """
    Generate an Excel workbook with:
      Sheet 1 – Full attendance log
      Sheet 2 – Daily summary (present / late / absent counts)
      Sheet 3 – Per-person summary
    """
    pd = _try_pandas()
    if pd is None:
        raise ImportError("pandas required.")

    filename = filename or f"summary_{_timestamp()}.xlsx"
    path = os.path.join(EXPORT_DIR, filename)

    df = pd.DataFrame(records)
    if df.empty:
        logger.warning("No records to summarise.")
        return export_attendance_excel(records, filename)

    # Sheet 1 – raw log
    cols_order = ["date", "name", "employee_id", "department",
                  "check_in", "check_out", "status", "confidence"]
    present_cols = [c for c in cols_order if c in df.columns]
    df_log = df[present_cols].copy()

    # Sheet 2 – daily summary
    if "date" in df.columns and "status" in df.columns:
        daily = df.groupby(["date", "status"]).size().unstack(fill_value=0).reset_index()
    else:
        daily = pd.DataFrame()

    # Sheet 3 – per-person
    if "name" in df.columns:
        per_person = df.groupby(["name", "employee_id"]).agg(
            total_days=("date", "count"),
            present=("status", lambda x: (x == "present").sum()),
            late=("status",    lambda x: (x == "late").sum()),
        ).reset_index()
    else:
        per_person = pd.DataFrame()

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df_log.to_excel(writer,     sheet_name="Attendance Log",    index=False)
        if not daily.empty:
            daily.to_excel(writer,      sheet_name="Daily Summary",      index=False)
        if not per_person.empty:
            per_person.to_excel(writer, sheet_name="Per-Person Summary", index=False)

        for sheet_name in writer.sheets:
            ws = writer.sheets[sheet_name]
            for col in ws.columns:
                max_len = max((len(str(c.value)) for c in col if c.value), default=10)
                ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 40)

    logger.info(f"Summary workbook → {path}")
    return path
