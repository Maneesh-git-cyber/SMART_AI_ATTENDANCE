#!/usr/bin/env python3
"""
setup.py  — One-time setup for the AI Attendance System
Installs all dependencies and verifies them.
"""

import subprocess
import sys
import os


def run(cmd, label):
    print(f"\n[SETUP] {label}…")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"[WARN] Command failed (may be OK if already installed): {cmd}")
    return result.returncode == 0


def main():
    print("=" * 60)
    print("  AI Attendance System — Setup")
    print("=" * 60)

    # 1. Upgrade pip
    run(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip")

    # 2. Core packages
    packages = [
        "opencv-python>=4.8.0",
        "opencv-contrib-python>=4.8.0",
        "numpy>=1.24.0",
        "mtcnn>=0.1.1",
        "tensorflow>=2.13.0",
        "keras-facenet>=0.3.2",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "Pillow>=10.0.0",
        "pandas>=2.0.0",
        "openpyxl>=3.1.2",
        "imutils>=0.5.4",
    ]
    run(
        f"{sys.executable} -m pip install " + " ".join(f'"{p}"' for p in packages),
        "Installing dependencies"
    )

    # 3. Create directories
    base = os.path.dirname(os.path.abspath(__file__))
    for d in ["database", "exports", "logs", "registered_faces", "src"]:
        os.makedirs(os.path.join(base, d), exist_ok=True)
    print("[SETUP] Directories created.")

    # 4. Verify imports
    print("\n[SETUP] Verifying imports…")
    checks = [
        ("cv2",            "OpenCV"),
        ("numpy",          "NumPy"),
        ("mtcnn",          "MTCNN"),
        ("keras_facenet",  "FaceNet (keras-facenet)"),
        ("pandas",         "Pandas"),
        ("openpyxl",       "openpyxl"),
        ("PIL",            "Pillow"),
    ]
    ok = True
    for mod, label in checks:
        try:
            __import__(mod)
            print(f"  ✓  {label}")
        except ImportError as e:
            print(f"  ✗  {label} — {e}")
            ok = False

    print()
    if ok:
        print("[SETUP] ✓ All dependencies installed successfully!")
        print()
        print("Next steps:")
        print("  1.  python main.py enroll          # Register persons")
        print("  2.  python main.py run             # Start live attendance")
        print("  OR")
        print("  1.  python gui_dashboard.py        # Launch the GUI")
    else:
        print("[SETUP] ✗ Some dependencies failed. Check the messages above.")
        print("        Try running:  pip install -r requirements.txt")

    print("=" * 60)


if __name__ == "__main__":
    main()
