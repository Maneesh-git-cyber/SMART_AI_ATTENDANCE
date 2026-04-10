"""
app.py — Flask REST API + WebSocket-free live-feed backend
for the AI Attendance System
"""

import os
import sys
import cv2
import numpy as np
import base64
import threading
import time
import json
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from core.database import (
    init_db, add_person, get_all_persons, get_person, update_person,
    delete_person, save_embedding, count_embeddings, get_embeddings_for_person,
    create_session, end_session, get_active_session, get_all_sessions,
    mark_attendance, get_attendance_for_session, get_today_attendance,
    is_marked_today, get_attendance_stats, log_spoof_attempt,
)
from core.face_engine import FaceRecognizer, detect_faces, extract_embedding, draw_face_box
from core.anti_spoofing import LivenessChecker
from core.export_utils import (
    export_attendance_csv, export_attendance_excel, export_persons_csv,
    export_all_sessions_excel, import_persons_from_csv, list_exports,
)

# ── Init ──────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)
init_db()

recognizer = FaceRecognizer()
recognizer.load_from_db()

REGISTERED_FACES_DIR = Path(__file__).parent / "registered_faces"
REGISTERED_FACES_DIR.mkdir(exist_ok=True)
LIVENESS_THRESHOLD = 60     # 0–100 scale: liveness_score must be >= 60

# ── Camera State (shared) ─────────────────────────────────────────────────────
class CameraState:
    def __init__(self):
        self.cap: cv2.VideoCapture | None = None
        self.running = False
        self.current_frame: np.ndarray | None = None
        self.detection_results: list = []
        self.liveness_checkers: dict[int, LivenessChecker] = {}  # face_idx -> checker
        self.session_id: int | None = None
        self.lock = threading.Lock()
        self.marked_this_session: set = set()
        self.frame_count: int = 0

cam = CameraState()


def camera_loop():
    """Background thread: reads camera, runs detection + recognition."""
    while cam.running:
        ret, frame = cam.cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        results = []
        cam.frame_count += 1
        try:
            faces = detect_faces(frame)

            for idx, face_data in enumerate(faces):
                box = face_data["box"]
                prob = face_data["prob"]
                tensor = face_data["face_tensor"]

                # Liveness
                if idx not in cam.liveness_checkers:
                    cam.liveness_checkers[idx] = LivenessChecker()
                liveness = cam.liveness_checkers[idx].update(frame, box)
                liveness_score = liveness["liveness_score"]
                is_live = bool(liveness.get('is_live', False))

                person_id = None
                person_name = "Unknown"
                confidence = 0.0
                color = (0, 0, 200)   # red = unknown/spoof
                marked = False

                if is_live and tensor is not None:
                    emb = extract_embedding(tensor)
                    person_id, confidence = recognizer.identify(emb)
                    if person_id:
                        p = get_person(person_id)
                        person_name = p["name"] if p else "Unknown"
                        color = (0, 220, 0)  # green = recognised

                        # Mark attendance
                        if (cam.session_id and
                                person_id not in cam.marked_this_session):
                            newly = mark_attendance(person_id, cam.session_id,
                                                    confidence, liveness_score)
                            if newly:
                                cam.marked_this_session.add(person_id)
                                marked = True
                    else:
                        color = (0, 165, 255)  # orange = live but unknown

                elif not is_live:
                    color = (0, 0, 200)
                    # Throttled spoof logging — once per 90 frames
                    if idx == 0 and cam.running and cam.frame_count % 90 == 0:
                        status_msg = cam.liveness_checkers[idx].get_status_message()
                        log_spoof_attempt(reason=f"Score={liveness_score:.2f} | {status_msg}")

                # Show liveness status message on frame
                status_msg = cam.liveness_checkers[idx].get_status_message() if not is_live else ""
                draw_face_box(frame, box, person_name, confidence,
                              liveness_score, color)
                if status_msg and not is_live:
                    x2, y2, w2, h2 = box
                    cv2.putText(frame, status_msg, (x2, y2 + h2 + 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 120, 255), 1, cv2.LINE_AA)

                results.append({
                    "box": list(box),
                    "person_id": person_id,
                    "name": person_name,
                    "confidence": round(confidence, 3),
                    "liveness_score": liveness_score,
                    "photo_score": liveness.get("photo_score", 0),
                    "is_live": is_live,
                    "marked": marked,
                    "status_msg": status_msg,
                })

        except Exception as e:
            pass

        # Overlay info
        active = get_active_session()
        if active:
            cv2.putText(frame, f"Session: {active['name']}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 100), 2)
        stats = get_attendance_stats(cam.session_id)
        cv2.putText(frame,
                    f"Present: {stats['present']} / {stats['total']}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
        ts = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, ts, (frame.shape[1] - 100, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        with cam.lock:
            cam.current_frame = frame.copy()
            cam.detection_results = results

        time.sleep(0.03)  # ~30 fps


# ── Camera Endpoints ──────────────────────────────────────────────────────────

@app.route("/api/camera/start", methods=["POST"])
def start_camera():
    data = request.json or {}
    cam_index = int(data.get("camera_index", 0))
    if cam.running:
        return jsonify({"status": "already_running"})
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        return jsonify({"error": "Cannot open camera"}), 400
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cam.cap = cap
    cam.running = True
    cam.liveness_checkers.clear()

    active = get_active_session()
    cam.session_id = active["id"] if active else None
    cam.marked_this_session = set()

    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()
    return jsonify({"status": "started", "session_id": cam.session_id})


@app.route("/api/camera/stop", methods=["POST"])
def stop_camera():
    cam.running = False
    time.sleep(0.3)
    if cam.cap:
        cam.cap.release()
        cam.cap = None
    cam.current_frame = None
    cam.liveness_checkers.clear()
    return jsonify({"status": "stopped"})


@app.route("/api/camera/frame")
def get_frame():
    """MJPEG-like single frame as base64 JSON for polling."""
    with cam.lock:
        frame = cam.current_frame
        results = cam.detection_results[:]
    if frame is None:
        return jsonify({"frame": None, "detections": []})
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    b64 = base64.b64encode(buf).decode()
    return jsonify({"frame": b64, "detections": results})


@app.route("/api/camera/stream")
def video_stream():
    """MJPEG stream endpoint."""
    def generate():
        while True:
            with cam.lock:
                frame = cam.current_frame
            if frame is None:
                time.sleep(0.05)
                continue
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
                   buf.tobytes() + b"\r\n")
            time.sleep(0.04)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/camera/status")
def camera_status():
    return jsonify({
        "running": cam.running,
        "session_id": cam.session_id,
        "faces_detected": len(cam.detection_results),
    })


# ── Persons ───────────────────────────────────────────────────────────────────

@app.route("/api/persons", methods=["GET"])
def list_persons():
    return jsonify(get_all_persons())


@app.route("/api/persons", methods=["POST"])
def create_person():
    d = request.json or {}
    name = d.get("name", "").strip()
    emp_id = d.get("employee_id", "").strip()
    if not name or not emp_id:
        return jsonify({"error": "name and employee_id required"}), 400
    try:
        pid = add_person(name, emp_id, d.get("department",""), d.get("email",""))
        return jsonify({"id": pid, "name": name, "employee_id": emp_id}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 409


@app.route("/api/persons/<int:pid>", methods=["GET"])
def get_person_ep(pid):
    p = get_person(pid)
    if not p:
        return jsonify({"error": "not found"}), 404
    p["embedding_count"] = count_embeddings(pid)
    return jsonify(p)


@app.route("/api/persons/<int:pid>", methods=["PUT"])
def update_person_ep(pid):
    d = request.json or {}
    update_person(pid, **d)
    return jsonify({"status": "updated"})


@app.route("/api/persons/<int:pid>", methods=["DELETE"])
def delete_person_ep(pid):
    delete_person(pid)
    recognizer.load_from_db()
    return jsonify({"status": "deleted"})


# ── Face Registration ─────────────────────────────────────────────────────────

@app.route("/api/persons/<int:pid>/register", methods=["POST"])
def register_face(pid):
    """
    Register face samples for a person.
    Expects multipart/form-data with field 'image' (JPEG/PNG).
    Or JSON { "use_camera": true } to grab current frame.
    """
    p = get_person(pid)
    if not p:
        return jsonify({"error": "Person not found"}), 404

    img_bgr = None

    if request.content_type and "multipart" in request.content_type:
        f = request.files.get("image")
        if not f:
            return jsonify({"error": "No image"}), 400
        buf = np.frombuffer(f.read(), dtype=np.uint8)
        img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    else:
        d = request.json or {}
        if d.get("use_camera"):
            with cam.lock:
                img_bgr = cam.current_frame
            if img_bgr is None:
                return jsonify({"error": "Camera not active"}), 400
            img_bgr = img_bgr.copy()
        elif d.get("image_b64"):
            raw = base64.b64decode(d["image_b64"])
            buf = np.frombuffer(raw, dtype=np.uint8)
            img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return jsonify({"error": "Could not decode image"}), 400

    faces = detect_faces(img_bgr)
    if not faces:
        return jsonify({"error": "No face detected in image"}), 422

    # Use highest-confidence face
    face = max(faces, key=lambda x: x["prob"])
    if face["face_tensor"] is None:
        return jsonify({"error": "Could not extract face tensor"}), 422

    emb = extract_embedding(face["face_tensor"])
    quality = float(face["prob"])
    save_embedding(pid, emb, quality)
    recognizer.add_embedding(pid, emb)

    # Save thumbnail
    x, y, w, h = face["box"]
    thumb = img_bgr[y:y+h, x:x+w]
    if thumb.size > 0:
        thumb_path = REGISTERED_FACES_DIR / f"{p['employee_id']}_latest.jpg"
        cv2.imwrite(str(thumb_path), cv2.resize(thumb, (128, 128)))
        update_person(pid, photo_path=str(thumb_path))

    n = count_embeddings(pid)
    return jsonify({
        "status": "registered",
        "person_id": pid,
        "embedding_count": n,
        "quality": round(quality, 3),
    })


@app.route("/api/persons/<int:pid>/photo")
def person_photo(pid):
    p = get_person(pid)
    if not p or not p.get("photo_path"):
        return jsonify({"error": "no photo"}), 404
    path = p["photo_path"]
    if not os.path.exists(path):
        return jsonify({"error": "file not found"}), 404
    return send_file(path, mimetype="image/jpeg")


# ── Sessions ──────────────────────────────────────────────────────────────────

@app.route("/api/sessions", methods=["GET"])
def list_sessions():
    return jsonify(get_all_sessions())


@app.route("/api/sessions", methods=["POST"])
def new_session():
    d = request.json or {}
    name = d.get("name", f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    sid = create_session(name)
    cam.session_id = sid
    cam.marked_this_session = set()
    return jsonify({"id": sid, "name": name}), 201


@app.route("/api/sessions/active", methods=["GET"])
def active_session():
    s = get_active_session()
    if not s:
        return jsonify(None)
    s["stats"] = get_attendance_stats(s["id"])
    return jsonify(s)


@app.route("/api/sessions/<int:sid>/end", methods=["POST"])
def close_session(sid):
    end_session(sid)
    if cam.session_id == sid:
        cam.session_id = None
    return jsonify({"status": "ended"})


# ── Attendance ────────────────────────────────────────────────────────────────

@app.route("/api/attendance/today", methods=["GET"])
def today_attendance():
    return jsonify(get_today_attendance())


@app.route("/api/attendance/session/<int:sid>", methods=["GET"])
def session_attendance(sid):
    records = get_attendance_for_session(sid)
    return jsonify(records)


@app.route("/api/attendance/stats", methods=["GET"])
def stats():
    sid = request.args.get("session_id")
    s = get_attendance_stats(int(sid) if sid else None)
    return jsonify(s)


@app.route("/api/attendance/live", methods=["GET"])
def live_detections():
    with cam.lock:
        return jsonify(cam.detection_results)


# ── Export / Import ───────────────────────────────────────────────────────────

@app.route("/api/export/attendance/csv", methods=["GET"])
def export_att_csv():
    sid = request.args.get("session_id")
    if sid:
        records = get_attendance_for_session(int(sid))
        sess = get_active_session()
        sname = sess["name"] if sess else str(sid)
    else:
        records = get_today_attendance()
        sname = "today"
    path = export_attendance_csv(records, sname)
    return send_file(path, as_attachment=True)


@app.route("/api/export/attendance/excel", methods=["GET"])
def export_att_excel():
    sid = request.args.get("session_id")
    if sid:
        records = get_attendance_for_session(int(sid))
        sess_row = [s for s in get_all_sessions() if s["id"] == int(sid)]
        sname = sess_row[0]["name"] if sess_row else str(sid)
    else:
        records = get_today_attendance()
        sname = "today"
    path = export_attendance_excel(records, sname)
    return send_file(path, as_attachment=True)


@app.route("/api/export/persons/csv", methods=["GET"])
def export_persons():
    persons = get_all_persons()
    path = export_persons_csv(persons)
    return send_file(path, as_attachment=True)


@app.route("/api/export/full", methods=["GET"])
def export_full():
    sessions = get_all_sessions()
    all_att = []
    for s in sessions:
        records = get_attendance_for_session(s["id"])
        for r in records:
            r["session_name"] = s["name"]
        all_att.extend(records)
    path = export_all_sessions_excel(sessions, all_att)
    return send_file(path, as_attachment=True)


@app.route("/api/import/persons", methods=["POST"])
def import_persons():
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "No file"}), 400
    tmp = Path("/tmp") / f.filename
    f.save(str(tmp))
    try:
        persons_data = import_persons_from_csv(str(tmp))
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    added = 0
    skipped = 0
    for pd_row in persons_data:
        try:
            add_person(
                str(pd_row.get("name","")),
                str(pd_row.get("employee_id","")),
                str(pd_row.get("department","")),
                str(pd_row.get("email","")),
            )
            added += 1
        except Exception:
            skipped += 1
    recognizer.load_from_db()
    return jsonify({"added": added, "skipped": skipped})


@app.route("/api/exports/list", methods=["GET"])
def list_export_files():
    return jsonify(list_exports())


# ── Reload embeddings ─────────────────────────────────────────────────────────

@app.route("/api/recognizer/reload", methods=["POST"])
def reload_recognizer():
    recognizer.load_from_db()
    return jsonify({"embeddings": len(recognizer)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
