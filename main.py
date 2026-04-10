"""
main.py
Command-line interface for the AI Attendance System.

Usage:
  python main.py run           # Start live attendance marking
  python main.py enroll        # Register a new person
  python main.py list          # List registered persons
  python main.py export        # Export today's attendance
  python main.py export-range  # Export attendance for a date range
  python main.py delete        # Soft-delete a person
  python main.py settings      # View / update system settings
  python main.py template      # Generate CSV enrolment template
"""

import sys
import os
import argparse
import logging
from datetime import date, timedelta

# ── Path fix so imports work from project root ────────────────────────────────
SRC = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, SRC)

import database_manager as db
from exporter import (export_attendance_csv, export_attendance_excel,
                      generate_summary_excel, export_persons_csv,
                      generate_persons_template)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(__file__),
                                         "logs", "system.log")),
    ],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────

def cmd_run(args):
    """Start the live attendance system."""
    from attendance_engine import AttendanceEngine
    thresh = float(db.get_setting("similarity_threshold") or 0.68)
    engine = AttendanceEngine(
        camera_id=args.camera,
        similarity_threshold=thresh,
        show_window=True,
    )
    try:
        engine.run()
    except KeyboardInterrupt:
        engine.stop()
        print("\n[INFO] Attendance session ended.")


def cmd_enroll(args):
    """Interactively enroll a new person."""
    from registration import enroll_person

    print("=== New Person Enrollment ===")
    name        = input("Full name        : ").strip()
    employee_id = input("Employee / Roll ID: ").strip()
    department  = input("Department        : ").strip()
    email       = input("Email (optional)  : ").strip()
    phone       = input("Phone (optional)  : ").strip()

    if not name or not employee_id:
        print("[ERROR] Name and employee ID are required.")
        return

    pid = enroll_person(
        name=name,
        employee_id=employee_id,
        department=department,
        email=email,
        phone=phone,
        camera_id=args.camera,
    )
    if pid:
        print(f"[OK] Enrolled successfully (DB id = {pid}).")
    else:
        print("[FAIL] Enrollment failed. Check the log.")


def cmd_list(args):
    """Print all registered persons."""
    persons = db.get_all_persons()
    if not persons:
        print("No registered persons found.")
        return
    fmt = "{:<5} {:<25} {:<15} {:<20} {}"
    print(fmt.format("ID", "Name", "Employee ID", "Department", "Registered At"))
    print("-" * 85)
    for p in persons:
        print(fmt.format(
            p["id"], p["name"][:24], p["employee_id"][:14],
            p.get("department", "")[:19],
            p.get("registered_at", "")[:19]
        ))
    print(f"\nTotal: {len(persons)} active persons.")


def cmd_export(args):
    """Export today's attendance."""
    records = db.get_today_attendance()
    if not records:
        print(f"No attendance records for today ({date.today()}).")
        return

    fmt = args.format.lower()
    if fmt == "excel":
        path = export_attendance_excel(records)
    elif fmt == "summary":
        path = generate_summary_excel(records)
    else:
        path = export_attendance_csv(records)

    print(f"[OK] Exported {len(records)} records → {path}")


def cmd_export_range(args):
    """Export attendance for a date range."""
    start = args.start or (date.today() - timedelta(days=7)).isoformat()
    end   = args.end   or date.today().isoformat()
    records = db.get_attendance_range(start, end)
    if not records:
        print(f"No records between {start} and {end}.")
        return

    fmt = args.format.lower()
    if fmt == "excel":
        path = export_attendance_excel(records)
    elif fmt == "summary":
        path = generate_summary_excel(records)
    else:
        path = export_attendance_csv(records)

    print(f"[OK] {len(records)} records ({start} → {end}) → {path}")


def cmd_delete(args):
    """Soft-delete a person by employee ID."""
    employee_id = args.employee_id or input("Employee ID to delete: ").strip()
    person = db.get_person_by_employee_id(employee_id)
    if not person:
        print(f"[ERROR] No active person with ID {employee_id!r}.")
        return
    confirm = input(f"Delete {person['name']} ({employee_id})? [y/N] ")
    if confirm.lower() == "y":
        db.delete_person(person["id"])
        print(f"[OK] {person['name']} deactivated.")
    else:
        print("Cancelled.")


def cmd_settings(args):
    """View / update system settings."""
    keys = [
        "similarity_threshold",
        "late_cutoff_time",
        "work_start_time",
        "spoof_blur_threshold",
        "spoof_lbp_threshold",
        "spoof_texture_threshold",
    ]
    print("=== System Settings ===")
    for k in keys:
        print(f"  {k:<35} = {db.get_setting(k)}")

    if args.set:
        for kv in args.set:
            if "=" not in kv:
                print(f"[WARN] Bad format: {kv!r} (use key=value)")
                continue
            k, v = kv.split("=", 1)
            db.update_setting(k.strip(), v.strip())
            print(f"  Updated: {k.strip()} = {v.strip()}")


def cmd_template(args):
    path = generate_persons_template()
    print(f"[OK] Template CSV → {path}")


# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="attendance",
        description="AI Attendance System — MTCNN + FaceNet + SQLite"
    )
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device index (default: 0)")

    sub = parser.add_subparsers(dest="command", required=True)

    # run
    sub.add_parser("run", help="Start live attendance marking")

    # enroll
    sub.add_parser("enroll", help="Register a new person via webcam")

    # list
    sub.add_parser("list", help="List all registered persons")

    # export (today)
    p_exp = sub.add_parser("export", help="Export today's attendance")
    p_exp.add_argument("--format", default="csv",
                       choices=["csv", "excel", "summary"],
                       help="Output format")

    # export-range
    p_range = sub.add_parser("export-range", help="Export attendance for a date range")
    p_range.add_argument("--start", help="Start date YYYY-MM-DD")
    p_range.add_argument("--end",   help="End date YYYY-MM-DD")
    p_range.add_argument("--format", default="csv",
                         choices=["csv", "excel", "summary"])

    # delete
    p_del = sub.add_parser("delete", help="Deactivate a registered person")
    p_del.add_argument("--employee-id", dest="employee_id", help="Employee ID")

    # settings
    p_set = sub.add_parser("settings", help="View / update settings")
    p_set.add_argument("--set", nargs="+", metavar="KEY=VALUE",
                       help="Update settings, e.g. --set similarity_threshold=0.72")

    # template
    sub.add_parser("template", help="Generate persons enrolment CSV template")

    return parser


def main():
    # Ensure directories exist
    for d in ["database", "exports", "logs", "registered_faces"]:
        os.makedirs(os.path.join(os.path.dirname(__file__), d), exist_ok=True)

    db.initialize_database()

    parser = build_parser()
    args   = parser.parse_args()

    dispatch = {
        "run":          cmd_run,
        "enroll":       cmd_enroll,
        "list":         cmd_list,
        "export":       cmd_export,
        "export-range": cmd_export_range,
        "delete":       cmd_delete,
        "settings":     cmd_settings,
        "template":     cmd_template,
    }

    handler = dispatch.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
