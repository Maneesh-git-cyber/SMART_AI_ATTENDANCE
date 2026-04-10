"""
gui_dashboard.py
Optional Tkinter GUI for the AI Attendance System.
Provides:
  - Live camera view with annotations
  - Today's attendance table (auto-refresh)
  - Enrolment wizard
  - Export buttons (CSV / Excel / Summary)
  - Settings panel

Run:  python gui_dashboard.py
"""

import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
import threading
import queue
import time
import cv2
import numpy as np
from PIL import Image, ImageTk
from datetime import date

SRC = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, SRC)

import database_manager as db
from attendance_engine import AttendanceEngine
from exporter import (export_attendance_csv, export_attendance_excel,
                      generate_summary_excel)

# ─── Colors / fonts ──────────────────────────────────────────────────────────
BG      = "#0f1117"
PANEL   = "#1a1d27"
ACCENT  = "#00c896"
TEXT    = "#e8eaf0"
DIM     = "#6b7280"
RED     = "#ef4444"
YELLOW  = "#f59e0b"
FONT    = ("Segoe UI", 10)
FONT_B  = ("Segoe UI", 10, "bold")
FONT_LG = ("Segoe UI", 14, "bold")


class AttendanceDashboard(tk.Tk):

    def __init__(self):
        super().__init__()
        db.initialize_database()

        self.title("AI Attendance System")
        self.configure(bg=BG)
        self.geometry("1380x780")
        self.minsize(1100, 650)

        self._engine: AttendanceEngine | None = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self._camera_running = False

        self._build_ui()
        self._refresh_table()
        self._poll_frame()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        # Top bar
        bar = tk.Frame(self, bg=PANEL, height=52)
        bar.pack(fill="x", side="top")
        tk.Label(bar, text="⬡  AI Attendance System", bg=PANEL,
                 fg=ACCENT, font=("Segoe UI", 16, "bold")).pack(side="left", padx=18)
        today = date.today().strftime("%A, %d %B %Y")
        tk.Label(bar, text=today, bg=PANEL, fg=DIM, font=FONT).pack(side="right", padx=18)

        # Main layout
        main = tk.Frame(self, bg=BG)
        main.pack(fill="both", expand=True, padx=10, pady=8)

        # Left: camera
        left = tk.Frame(main, bg=BG)
        left.pack(side="left", fill="both", expand=True)
        self._build_camera_panel(left)

        # Right: controls + table
        right = tk.Frame(main, bg=BG, width=460)
        right.pack(side="right", fill="both", padx=(10, 0))
        right.pack_propagate(False)
        self._build_controls(right)
        self._build_table(right)

    def _build_camera_panel(self, parent):
        frame = tk.Frame(parent, bg=PANEL, bd=0, highlightthickness=1,
                         highlightbackground=ACCENT)
        frame.pack(fill="both", expand=True)

        tk.Label(frame, text="Live Camera Feed", bg=PANEL, fg=ACCENT,
                 font=FONT_B).pack(anchor="w", padx=10, pady=(8, 0))

        self._cam_label = tk.Label(frame, bg="#000000", text="Camera not started",
                                   fg=DIM, font=("Segoe UI", 12))
        self._cam_label.pack(fill="both", expand=True, padx=8, pady=8)

        # Camera controls
        ctrl = tk.Frame(frame, bg=PANEL)
        ctrl.pack(fill="x", padx=8, pady=(0, 8))

        self._cam_btn = tk.Button(ctrl, text="▶  Start Camera",
                                  command=self._toggle_camera,
                                  bg=ACCENT, fg="#000", font=FONT_B,
                                  relief="flat", padx=14, pady=6, cursor="hand2")
        self._cam_btn.pack(side="left")

        self._status_var = tk.StringVar(value="Stopped")
        tk.Label(ctrl, textvariable=self._status_var, bg=PANEL,
                 fg=DIM, font=FONT).pack(side="left", padx=12)

        self._fps_var = tk.StringVar(value="FPS: --")
        tk.Label(ctrl, textvariable=self._fps_var, bg=PANEL,
                 fg=DIM, font=FONT).pack(side="right")

    def _build_controls(self, parent):
        frame = tk.Frame(parent, bg=PANEL, bd=0, highlightthickness=1,
                         highlightbackground="#2d3148")
        frame.pack(fill="x", pady=(0, 8))

        tk.Label(frame, text="Actions", bg=PANEL, fg=ACCENT,
                 font=FONT_B).pack(anchor="w", padx=10, pady=(8, 6))

        btn_cfg = dict(bg="#252838", fg=TEXT, font=FONT, relief="flat",
                       padx=10, pady=7, cursor="hand2", anchor="w", width=20)
        btns = [
            ("👤  Enroll New Person",    self._enroll_dialog),
            ("📋  Export CSV",           self._export_csv),
            ("📊  Export Excel",         self._export_excel),
            ("📈  Export Summary",       self._export_summary),
            ("⚙   Settings",            self._settings_dialog),
            ("🔄  Refresh Table",        self._refresh_table),
        ]
        btn_row = tk.Frame(frame, bg=PANEL)
        btn_row.pack(fill="x", padx=10, pady=(0, 8))
        for i, (label, cmd) in enumerate(btns):
            r, c = divmod(i, 2)
            b = tk.Button(btn_row, text=label, command=cmd, **btn_cfg)
            b.grid(row=r, column=c, padx=3, pady=3, sticky="ew")
        btn_row.columnconfigure(0, weight=1)
        btn_row.columnconfigure(1, weight=1)

        # Stat row
        stat = tk.Frame(frame, bg=PANEL)
        stat.pack(fill="x", padx=10, pady=(0, 8))
        self._stat_marked = tk.StringVar(value="Marked today: 0")
        self._stat_total  = tk.StringVar(value="Registered: 0")
        tk.Label(stat, textvariable=self._stat_marked, bg=PANEL,
                 fg=ACCENT, font=FONT_B).pack(side="left")
        tk.Label(stat, textvariable=self._stat_total, bg=PANEL,
                 fg=DIM, font=FONT).pack(side="right")

    def _build_table(self, parent):
        frame = tk.Frame(parent, bg=PANEL, bd=0, highlightthickness=1,
                         highlightbackground="#2d3148")
        frame.pack(fill="both", expand=True)

        tk.Label(frame, text="Today's Attendance", bg=PANEL, fg=ACCENT,
                 font=FONT_B).pack(anchor="w", padx=10, pady=(8, 4))

        cols = ("Name", "Emp ID", "Dept", "Check-In", "Status", "Score")
        self._tree = ttk.Treeview(frame, columns=cols, show="headings",
                                  height=20)
        widths = [160, 90, 110, 90, 70, 55]
        for col, w in zip(cols, widths):
            self._tree.heading(col, text=col)
            self._tree.column(col, width=w, anchor="center")

        # Style
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", background=PANEL, foreground=TEXT,
                        rowheight=26, fieldbackground=PANEL,
                        borderwidth=0, font=FONT)
        style.configure("Treeview.Heading", background="#252838",
                        foreground=ACCENT, font=FONT_B)
        style.map("Treeview", background=[("selected", "#2a3550")])

        self._tree.tag_configure("present", foreground="#4ade80")
        self._tree.tag_configure("late",    foreground=YELLOW)

        sb = ttk.Scrollbar(frame, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y", padx=(0, 4))
        self._tree.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    # ── Camera ───────────────────────────────────────────────────────────────

    def _toggle_camera(self):
        if self._camera_running:
            self._stop_camera()
        else:
            self._start_camera()

    def _start_camera(self):
        thresh = float(db.get_setting("similarity_threshold") or 0.68)

        def engine_thread():
            self._engine = AttendanceEngine(
                similarity_threshold=thresh,
                show_window=False,           # GUI handles display
            )
            # Monkey-patch _running flag access
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                self._status_var.set("Camera error")
                return

            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            self._camera_running = True
            self._status_var.set("Running…")
            self._cam_btn.config(text="■  Stop Camera", bg=RED)

            # Load gallery & today already-marked
            self._engine._refresh_gallery()
            for rec in db.get_today_attendance():
                self._engine._marked_today.add(rec["person_id"])

            while self._camera_running:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.03)
                    continue

                self._engine._refresh_gallery()
                display = frame.copy()
                faces   = self._engine._liveness and []

                try:
                    from face_engine import detect_faces, align_and_crop, get_embedding, find_best_match, draw_face_box
                    faces = detect_faces(frame)
                except Exception:
                    faces = []

                for face_data in faces:
                    if face_data["confidence"] < 0.95:
                        continue
                    box  = face_data["box"]
                    kpts = face_data["keypoints"]
                    fc   = align_and_crop(frame, box, kpts)
                    if fc is None or fc.size == 0:
                        continue

                    is_live, reason, _ = self._engine._liveness.check(
                        fc, face_box=box, frame_shape=frame.shape)

                    if not is_live:
                        db.log_spoof_attempt(reason)
                        draw_face_box(display, box, "SPOOF", 0.0, spoof=True)
                        continue

                    emb = get_embedding(fc)
                    person, score = find_best_match(
                        emb, self._engine._gallery, self._engine.threshold)

                    if person is None:
                        draw_face_box(display, box, "Unknown", score,
                                      color=(0, 140, 255))
                        continue

                    pid  = person["id"]
                    name = person["name"]
                    now  = time.time()
                    if (pid not in self._engine._marked_today and
                            (now - self._engine._cooldown.get(pid, 0)) > 10):
                        success, msg = db.mark_attendance(pid, score)
                        self._engine._cooldown[pid] = now
                        if success:
                            self._engine._marked_today.add(pid)
                            self.after(100, self._refresh_table)

                    color = (0, 220, 0) if pid in self._engine._marked_today \
                            else (50, 200, 255)
                    draw_face_box(display, box, name, score, color=color)

                self._engine._draw_hud(display)

                # Push frame
                try:
                    self._frame_queue.put_nowait(display)
                except queue.Full:
                    pass

                # FPS
                self._engine.fps_counter += 1
                if time.time() - self._engine.fps_time >= 1:
                    self._engine.current_fps = self._engine.fps_counter / (
                        time.time() - self._engine.fps_time)
                    self._engine.fps_counter = 0
                    self._engine.fps_time    = time.time()
                    self._fps_var.set(f"FPS: {self._engine.current_fps:.1f}")

            cap.release()

        threading.Thread(target=engine_thread, daemon=True).start()

    def _stop_camera(self):
        self._camera_running = False
        self._cam_btn.config(text="▶  Start Camera", bg=ACCENT)
        self._status_var.set("Stopped")
        self._fps_var.set("FPS: --")

    def _poll_frame(self):
        try:
            frame = self._frame_queue.get_nowait()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = self._cam_label.winfo_height(), self._cam_label.winfo_width()
            if h > 10 and w > 10:
                img = Image.fromarray(frame_rgb).resize((w, h), Image.LANCZOS)
            else:
                img = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(img)
            self._cam_label.config(image=photo, text="")
            self._cam_label.image = photo
        except queue.Empty:
            pass
        self.after(33, self._poll_frame)  # ~30 fps UI

    # ── Table ────────────────────────────────────────────────────────────────

    def _refresh_table(self):
        for row in self._tree.get_children():
            self._tree.delete(row)

        records = db.get_today_attendance()
        for r in records:
            score = f"{r.get('confidence', 0):.0%}"
            tag   = r.get("status", "present")
            self._tree.insert("", "end", values=(
                r.get("name", ""),
                r.get("employee_id", ""),
                r.get("department", ""),
                (r.get("check_in", "") or "")[-8:],
                r.get("status", ""),
                score,
            ), tags=(tag,))

        total   = len(db.get_all_persons())
        self._stat_marked.set(f"Marked today: {len(records)}")
        self._stat_total.set(f"Registered: {total}")

        self.after(15_000, self._refresh_table)  # auto-refresh every 15 s

    # ── Dialogs ──────────────────────────────────────────────────────────────

    def _enroll_dialog(self):
        win = tk.Toplevel(self)
        win.title("Enroll New Person")
        win.configure(bg=BG)
        win.geometry("400x320")
        win.resizable(False, False)

        fields = [("Full Name*",    "name"),
                  ("Employee/Roll ID*", "emp_id"),
                  ("Department",    "dept"),
                  ("Email",         "email"),
                  ("Phone",         "phone")]
        entries = {}
        for label, key in fields:
            row = tk.Frame(win, bg=BG)
            row.pack(fill="x", padx=20, pady=4)
            tk.Label(row, text=label, bg=BG, fg=TEXT, font=FONT,
                     width=18, anchor="w").pack(side="left")
            e = tk.Entry(row, bg=PANEL, fg=TEXT, font=FONT,
                         insertbackground=TEXT, relief="flat", width=22)
            e.pack(side="left")
            entries[key] = e

        def start_enroll():
            name   = entries["name"].get().strip()
            emp_id = entries["emp_id"].get().strip()
            if not name or not emp_id:
                messagebox.showerror("Error", "Name and Employee ID are required.",
                                     parent=win)
                return
            win.destroy()

            def enroll_thread():
                from registration import enroll_person
                pid = enroll_person(
                    name=name, employee_id=emp_id,
                    department=entries["dept"].get().strip(),
                    email=entries["email"].get().strip(),
                    phone=entries["phone"].get().strip(),
                )
                msg = f"Enrolled {name} (id={pid})" if pid else "Enrollment failed."
                self.after(0, lambda: messagebox.showinfo("Enrollment", msg))
                self.after(0, self._refresh_table)

            threading.Thread(target=enroll_thread, daemon=True).start()

        tk.Button(win, text="Start Enrollment (webcam)", command=start_enroll,
                  bg=ACCENT, fg="#000", font=FONT_B, relief="flat",
                  padx=12, pady=8, cursor="hand2").pack(pady=18)

    def _export_csv(self):
        records = db.get_today_attendance()
        if not records:
            messagebox.showinfo("Export", "No records for today.")
            return
        path = export_attendance_csv(records)
        messagebox.showinfo("Exported", f"CSV saved:\n{path}")

    def _export_excel(self):
        records = db.get_today_attendance()
        if not records:
            messagebox.showinfo("Export", "No records for today.")
            return
        path = export_attendance_excel(records)
        messagebox.showinfo("Exported", f"Excel saved:\n{path}")

    def _export_summary(self):
        from tkinter.simpledialog import askstring
        start = askstring("Date range", "Start date (YYYY-MM-DD, blank=today):",
                          parent=self)
        end   = askstring("Date range", "End date   (YYYY-MM-DD, blank=today):",
                          parent=self)
        today = date.today().isoformat()
        records = db.get_attendance_range(start or today, end or today)
        if not records:
            messagebox.showinfo("Export", "No records in range.")
            return
        path = generate_summary_excel(records)
        messagebox.showinfo("Exported", f"Summary Excel saved:\n{path}")

    def _settings_dialog(self):
        win = tk.Toplevel(self)
        win.title("Settings")
        win.configure(bg=BG)
        win.geometry("440x340")

        keys = [
            ("Similarity Threshold (0–1)", "similarity_threshold"),
            ("Work Start Time (HH:MM)",    "work_start_time"),
            ("Late Cutoff Time (HH:MM)",   "late_cutoff_time"),
            ("Spoof Blur Threshold",       "spoof_blur_threshold"),
            ("Spoof LBP Threshold",        "spoof_lbp_threshold"),
            ("Spoof Freq Threshold",       "spoof_texture_threshold"),
        ]
        entries = {}
        for label, key in keys:
            row = tk.Frame(win, bg=BG)
            row.pack(fill="x", padx=20, pady=5)
            tk.Label(row, text=label, bg=BG, fg=TEXT, font=FONT,
                     width=32, anchor="w").pack(side="left")
            e = tk.Entry(row, bg=PANEL, fg=TEXT, font=FONT,
                         insertbackground=TEXT, relief="flat", width=12)
            e.insert(0, db.get_setting(key) or "")
            e.pack(side="left")
            entries[key] = e

        def save():
            for key, entry in entries.items():
                val = entry.get().strip()
                if val:
                    db.update_setting(key, val)
            messagebox.showinfo("Settings", "Settings saved.", parent=win)
            win.destroy()

        tk.Button(win, text="Save", command=save, bg=ACCENT, fg="#000",
                  font=FONT_B, relief="flat", padx=16, pady=7,
                  cursor="hand2").pack(pady=14)

    # ── Close ─────────────────────────────────────────────────────────────────

    def _on_close(self):
        self._stop_camera()
        self.destroy()


if __name__ == "__main__":
    for d in ["database", "exports", "logs", "registered_faces"]:
        os.makedirs(os.path.join(os.path.dirname(__file__), d), exist_ok=True)
    app = AttendanceDashboard()
    app.mainloop()
