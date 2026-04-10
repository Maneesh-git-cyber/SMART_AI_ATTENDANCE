"""
attendance_engine.py
Real-time attendance marking from live webcam feed.

Flow per frame:
  1. MTCNN detects faces
  2. Each face cropped & aligned
  3. Anti-spoof liveness check  → reject if spoof
  4. FaceNet embedding extracted
  5. Cosine-similarity match against DB gallery
  6. If match ≥ threshold → mark attendance (DB deduplicates per day)
  7. Draw annotated overlay on frame
"""

import cv2
import numpy as np
import time
import threading
import logging
from typing import Dict, List, Optional, Set
from collections import defaultdict
from datetime import date

from face_engine import (detect_faces, align_and_crop,
                         get_embedding, find_best_match, draw_face_box)
from anti_spoof import LivenessChecker
import database_manager as db

logger = logging.getLogger(__name__)


# ── Per-track cooldown to avoid hammering the DB ─────────────────────────────
MARK_COOLDOWN_SEC   = 10    # seconds before re-trying to mark same person
UNKNOWN_DISPLAY_SEC = 3     # seconds to show "Unknown" label


class AttendanceEngine:
    """
    Runs a live webcam loop and marks attendance automatically.
    Thread-safe: can be stopped via stop().
    """

    def __init__(self, camera_id: int = 0,
                 similarity_threshold: float = 0.68,
                 show_window: bool = True):
        self.camera_id  = camera_id
        self.threshold  = similarity_threshold
        self.show_window = show_window

        self._running   = False
        self._thread: Optional[threading.Thread] = None
        self._lock      = threading.Lock()

        # Attendance state for current session
        self._marked_today: Set[int] = set()          # person_ids already marked
        self._cooldown: Dict[int, float] = {}          # person_id → last attempt time

        # Gallery (in-memory cache, reloaded every 60 s)
        self._gallery: List[Dict] = []
        self._gallery_loaded_at = 0.0

        self._liveness = LivenessChecker(
            blur_thresh=float(db.get_setting("spoof_blur_threshold")    or 80.0),
            lbp_thresh =float(db.get_setting("spoof_lbp_threshold")     or 12.0),
            freq_thresh=float(db.get_setting("spoof_texture_threshold") or 0.35),
        )

        # Metrics
        self.fps_counter = 0
        self.fps_time    = time.time()
        self.current_fps = 0.0

        # On-screen attendance log (last N events)
        self._log: List[str] = []
        self._MAX_LOG = 8

    # ── Gallery ───────────────────────────────────────────────────────────────

    def _refresh_gallery(self):
        now = time.time()
        if now - self._gallery_loaded_at > 60:
            self._gallery = db.get_all_persons()
            self._gallery_loaded_at = now
            logger.info(f"Gallery refreshed: {len(self._gallery)} persons.")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        """Blocking call. Opens camera and processes frames until stopped."""
        cap = cv2.VideoCapture(self.camera_id,
                               cv2.CAP_DSHOW if __import__("os").name == "nt"
                               else cv2.CAP_ANY)
        if not cap.isOpened():
            logger.error("Cannot open camera.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._running = True
        logger.info("Attendance engine started.")

        # Load today's already-marked persons to avoid re-marking on restart
        today_records = db.get_today_attendance()
        for rec in today_records:
            self._marked_today.add(rec["person_id"])

        while self._running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            self._refresh_gallery()
            display = frame.copy()
            faces   = detect_faces(frame)

            for face_data in faces:
                if face_data["confidence"] < 0.95:
                    continue

                box   = face_data["box"]
                kpts  = face_data["keypoints"]
                x, y, w, h = box

                face_crop = align_and_crop(frame, box, kpts)
                if face_crop is None or face_crop.size == 0:
                    continue

                # ── Liveness ─────────────────────────────────────────────
                is_live, reason, _ = self._liveness.check(
                    face_crop, face_box=box, frame_shape=frame.shape
                )

                if not is_live:
                    db.log_spoof_attempt(reason)
                    draw_face_box(display, box, "SPOOF", 0.0, spoof=True)
                    self._push_log(f"⚠ Spoof: {reason[:40]}")
                    continue

                # ── Embedding + Match ─────────────────────────────────────
                emb = get_embedding(face_crop)
                person, score = find_best_match(
                    emb, self._gallery, self.threshold
                )

                if person is None:
                    draw_face_box(display, box, "Unknown", score,
                                  color=(0, 140, 255))
                    continue

                pid  = person["id"]
                name = person["name"]

                # ── Mark attendance ───────────────────────────────────────
                now = time.time()
                already_marked = pid in self._marked_today
                in_cooldown    = (now - self._cooldown.get(pid, 0)) < MARK_COOLDOWN_SEC

                if not already_marked and not in_cooldown:
                    success, msg = db.mark_attendance(pid, score)
                    self._cooldown[pid] = now
                    if success:
                        self._marked_today.add(pid)
                        self._push_log(f"✓ {name} — {msg}")
                        logger.info(f"Marked: {name} | score={score:.3f}")
                    else:
                        # DB said duplicate (edge case)
                        self._marked_today.add(pid)

                # ── Draw ──────────────────────────────────────────────────
                color = (0, 220, 0) if pid in self._marked_today else (50, 200, 255)
                draw_face_box(display, box, name, score, color=color)

            # ── HUD ───────────────────────────────────────────────────────
            self._draw_hud(display)

            if self.show_window:
                cv2.imshow("AI Attendance System  [Q=quit  R=refresh]", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self._running = False
                elif key == ord("r"):
                    self._gallery_loaded_at = 0  # force reload

            # FPS
            self.fps_counter += 1
            if time.time() - self.fps_time >= 1.0:
                self.current_fps = self.fps_counter / (time.time() - self.fps_time)
                self.fps_counter = 0
                self.fps_time    = time.time()

        cap.release()
        cv2.destroyAllWindows()
        logger.info("Attendance engine stopped.")

    def stop(self):
        self._running = False

    def start_threaded(self):
        """Run the engine in a background thread."""
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()
        return self._thread

    # ── HUD helpers ──────────────────────────────────────────────────────────

    def _push_log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self._log.append(f"[{ts}] {msg}")
        if len(self._log) > self._MAX_LOG:
            self._log.pop(0)

    def _draw_hud(self, frame: np.ndarray):
        h, w = frame.shape[:2]

        # Semi-transparent top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 55), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        today_str = date.today().strftime("%d %b %Y")
        marked    = len(self._marked_today)
        total     = len(self._gallery)

        cv2.putText(frame, f"AI Attendance  |  {today_str}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
        cv2.putText(frame, f"Marked: {marked}/{total}  |  FPS: {self.current_fps:.1f}",
                    (w - 310, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 180), 2)

        # Event log (bottom-left)
        log_x, log_y = 10, h - 20 - (len(self._log) * 22)
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, log_y - 10),
                      (500, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay2, 0.5, frame, 0.5, 0, frame)
        for i, line in enumerate(self._log):
            color = (0, 230, 100) if "✓" in line else (0, 90, 230) if "⚠" in line \
                    else (200, 200, 200)
            cv2.putText(frame, line, (log_x, log_y + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
