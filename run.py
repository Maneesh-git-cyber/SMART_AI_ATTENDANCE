#!/usr/bin/env python3
"""
run.py — One-click launcher for the AI Attendance System
"""

import subprocess
import sys
import os
import webbrowser
import threading
import time
from pathlib import Path

BASE = Path(__file__).parent


def check_and_install():
    print("=" * 60)
    print("  AI Attendance System — Startup")
    print("=" * 60)
    req = BASE / "requirements.txt"
    print("\n[1/3] Installing dependencies...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-r", str(req), "-q"
    ])
    print("      ✓ Dependencies installed")


def start_server():
    print("\n[2/3] Starting Flask backend on http://localhost:5000 ...")
    os.chdir(str(BASE))
    proc = subprocess.Popen(
        [sys.executable, "app.py"],
        cwd=str(BASE),
    )
    return proc


def open_frontend():
    print("\n[3/3] Opening frontend...")
    time.sleep(2)
    frontend = BASE / "frontend" / "index.html"
    webbrowser.open(f"file://{frontend}")
    print(f"      ✓ Opened {frontend}")
    print("\n" + "=" * 60)
    print("  System is running!")
    print("  Backend : http://localhost:5000")
    print(f"  Frontend: file://{frontend}")
    print("  Press Ctrl+C to stop")
    print("=" * 60)


if __name__ == "__main__":
    check_and_install()
    proc = start_server()
    t = threading.Thread(target=open_frontend, daemon=True)
    t.start()
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        print("\n[STOP] System stopped.")
