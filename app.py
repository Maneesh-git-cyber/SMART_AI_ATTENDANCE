"""
app.py — Flask backend for AI Attendance System
BUGS FIXED in this version:
  1. session_id was only set at camera-start time. If session was created AFTER
     camera started, cam.session_id stayed None → mark_attendance never called.
     FIX: poll active session dynamically inside camera_loop every frame.

  2. marked_this_session was never cleared when a new session was created via
     the API, so even if session_id updated, the person was already in the set.
     FIX: clear marked_this_session whenever session_id changes.

  3. is_live check was comparing float liveness_score (0.0–1.0) to int threshold 60.
     FIX: use liveness.get('is_live') directly from the checker's own boolean.

  4. frame_count attribute added to CameraState but never incremented in loop.
     FIX: properly increment inside loop.
"""

import os, sys, cv2, numpy as np, base64, threading, time
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS

sys.path.insert(0, str(Path(__file__).parent))

from core.database import (
    init_db, add_person, get_all_persons, get_person, update_person,
    delete_person, save_embedding, count_embeddings,
    create_session, end_session, get_active_session, get_all_sessions,
    mark_attendance, get_attendance_for_session, get_today_attendance,
    get_attendance_stats, log_spoof_attempt,
)
from core.face_engine import FaceRecognizer, detect_faces, extract_embedding, draw_face_box
from core.anti_spoofing import LivenessChecker
from core.export_utils import (
    export_attendance_csv, export_attendance_excel, export_persons_csv,
    export_all_sessions_excel, import_persons_from_csv, list_exports,
)

app = Flask(__name__)
CORS(app)
init_db()

recognizer = FaceRecognizer()
recognizer.load_from_db()

REGISTERED_FACES_DIR = Path(__file__).parent / "registered_faces"
REGISTERED_FACES_DIR.mkdir(exist_ok=True)


# ── Camera state ──────────────────────────────────────────────────────────────

class CameraState:
    def __init__(self):
        self.cap              = None
        self.running          = False
        self.current_frame    = None
        self.detection_results = []
        self.liveness_checkers = {}
        self.session_id       = None   # dynamically refreshed in loop
        self.marked_this_session = set()
        self.lock             = threading.Lock()
        self.frame_count      = 0

cam = CameraState()


def camera_loop():
    while cam.running:
        ret, frame = cam.cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        cam.frame_count += 1
        results = []

        # ── BUG FIX 1: refresh session_id every frame ──────────────────────
        active_sess = get_active_session()
        new_sid = active_sess["id"] if active_sess else None

        # ── BUG FIX 2: clear marked set when session changes ───────────────
        if new_sid != cam.session_id:
            cam.session_id = new_sid
            cam.marked_this_session = set()

        try:
            faces = detect_faces(frame)

            for idx, face_data in enumerate(faces):
                box    = face_data["box"]
                prob   = face_data["prob"]
                tensor = face_data["face_tensor"]

                # Liveness check
                if idx not in cam.liveness_checkers:
                    cam.liveness_checkers[idx] = LivenessChecker()

                liveness      = cam.liveness_checkers[idx].update(frame, box)
                liveness_score = liveness.get("liveness_score", 0)
                photo_score    = liveness.get("photo_score", 100)

                # ── BUG FIX 3: use checker's own boolean, not float compare ─
                is_live = bool(liveness.get("is_live", False))

                person_id   = None
                person_name = "Unknown"
                confidence  = 0.0
                color       = (0, 0, 200)
                marked      = False
                status_msg  = cam.liveness_checkers[idx].get_status_message()

                if is_live and tensor is not None:
                    emb = extract_embedding(tensor)
                    person_id, confidence = recognizer.identify(emb)

                    if person_id:
                        p = get_person(person_id)
                        person_name = p["name"] if p else "Unknown"
                        color = (0, 220, 0)

                        # ── Mark attendance ──────────────────────────────────
                        # Require session, not already marked, confidence > 0.65
                        if (cam.session_id is not None
                                and person_id not in cam.marked_this_session
                                and confidence >= 0.65):

                            newly = mark_attendance(
                                person_id, cam.session_id,
                                confidence, liveness_score
                            )
                            if newly:
                                cam.marked_this_session.add(person_id)
                                marked = True
                                print(f"[ATTEND] Marked: {person_name} | "
                                      f"session={cam.session_id} | "
                                      f"conf={confidence:.2f} | live={liveness_score}")
                            else:
                                # Already marked — show tick on box still
                                cam.marked_this_session.add(person_id)

                    else:
                        color = (0, 165, 255)   # orange = live but unknown

                else:
                    color = (0, 0, 200)
                    if idx == 0 and cam.frame_count % 90 == 0:
                        log_spoof_attempt(
                            reason=f"Photo/spoof score={photo_score:.0f} | {status_msg}"
                        )

                draw_face_box(frame, box, person_name, confidence,
                              liveness_score / 100.0, color)

                if not is_live and status_msg:
                    bx, by, bw, bh = box
                    cv2.putText(frame, status_msg,
                                (bx, by + bh + 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                                (0, 100, 255), 1, cv2.LINE_AA)

                # Whether already-marked or newly marked
                already = person_id in cam.marked_this_session if person_id else False

                results.append({
                    "box":           list(box),
                    "person_id":     person_id,
                    "name":          person_name,
                    "confidence":    round(confidence, 3),
                    "liveness_score": round(liveness_score, 1),
                    "photo_score":   round(photo_score, 1),
                    "is_live":       is_live,
                    "marked":        marked or already,
                    "status_msg":    status_msg if not is_live else "",
                })

        except Exception as e:
            print(f"[LOOP ERR] {e}")

        # ── Overlay stats ──────────────────────────────────────────────────
        if active_sess:
            cv2.putText(frame, f"Session: {active_sess['name']}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 100), 2)
        stats = get_attendance_stats(cam.session_id)
        cv2.putText(frame,
                    f"Present: {stats['present']} / {stats['total']}",
                    (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
        cv2.putText(frame, datetime.now().strftime("%H:%M:%S"),
                    (frame.shape[1] - 110, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        with cam.lock:
            cam.current_frame    = frame.copy()
            cam.detection_results = results

        time.sleep(0.03)


# ── Camera endpoints ──────────────────────────────────────────────────────────

@app.route("/api/camera/start", methods=["POST"])
def start_camera():
    data      = request.json or {}
    cam_index = int(data.get("camera_index", 0))
    if cam.running:
        return jsonify({"status": "already_running"})
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        return jsonify({"error": "Cannot open camera"}), 400
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cam.cap     = cap
    cam.running = True
    cam.liveness_checkers.clear()
    cam.frame_count = 0

    # Pick up any already-active session
    active = get_active_session()
    cam.session_id = active["id"] if active else None
    cam.marked_this_session = set()

    threading.Thread(target=camera_loop, daemon=True).start()
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
    with cam.lock:
        frame   = cam.current_frame
        results = cam.detection_results[:]
    if frame is None:
        return jsonify({"frame": None, "detections": []})
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return jsonify({
        "frame":      base64.b64encode(buf).decode(),
        "detections": results,
        "session_id": cam.session_id,
    })


@app.route("/api/camera/stream")
def video_stream():
    def generate():
        while True:
            with cam.lock:
                frame = cam.current_frame
            if frame is None:
                time.sleep(0.05)
                continue
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + buf.tobytes() + b"\r\n")
            time.sleep(0.04)
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/camera/status")
def camera_status():
    return jsonify({
        "running":        cam.running,
        "session_id":     cam.session_id,
        "faces_detected": len(cam.detection_results),
    })


# ── Persons ───────────────────────────────────────────────────────────────────

@app.route("/api/persons", methods=["GET"])
def list_persons():
    return jsonify(get_all_persons())


@app.route("/api/persons", methods=["POST"])
def create_person():
    d = request.json or {}
    name   = d.get("name", "").strip()
    emp_id = d.get("employee_id", "").strip()
    if not name or not emp_id:
        return jsonify({"error": "name and employee_id required"}), 400
    try:
        pid = add_person(name, emp_id,
                         d.get("department", ""), d.get("email", ""))
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
    update_person(pid, **(request.json or {}))
    return jsonify({"status": "updated"})


@app.route("/api/persons/<int:pid>", methods=["DELETE"])
def delete_person_ep(pid):
    delete_person(pid)
    recognizer.load_from_db()
    return jsonify({"status": "deleted"})


# ── Face registration ─────────────────────────────────────────────────────────

@app.route("/api/persons/<int:pid>/register", methods=["POST"])
def register_face(pid):
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

    face = max(faces, key=lambda x: x["prob"])
    if face["face_tensor"] is None:
        return jsonify({"error": "Could not extract face tensor"}), 422

    emb     = extract_embedding(face["face_tensor"])
    quality = float(face["prob"])
    save_embedding(pid, emb, quality)
    recognizer.add_embedding(pid, emb)

    x, y, w, h = face["box"]
    thumb = img_bgr[y:y+h, x:x+w]
    if thumb.size > 0:
        thumb_path = REGISTERED_FACES_DIR / f"{p['employee_id']}_latest.jpg"
        cv2.imwrite(str(thumb_path), cv2.resize(thumb, (128, 128)))
        update_person(pid, photo_path=str(thumb_path))

    return jsonify({
        "status":          "registered",
        "person_id":       pid,
        "embedding_count": count_embeddings(pid),
        "quality":         round(quality, 3),
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
    d    = request.json or {}
    name = d.get("name",
                 f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    sid  = create_session(name)
    # ── BUG FIX: update cam state immediately when session created ─────────
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
        cam.marked_this_session = set()
    return jsonify({"status": "ended"})


# ── Attendance ────────────────────────────────────────────────────────────────

@app.route("/api/attendance/today", methods=["GET"])
def today_attendance():
    return jsonify(get_today_attendance())


@app.route("/api/attendance/session/<int:sid>", methods=["GET"])
def session_attendance(sid):
    return jsonify(get_attendance_for_session(sid))


@app.route("/api/attendance/stats", methods=["GET"])
def stats():
    sid = request.args.get("session_id")
    return jsonify(get_attendance_stats(int(sid) if sid else None))


@app.route("/api/attendance/live", methods=["GET"])
def live_detections():
    with cam.lock:
        return jsonify(cam.detection_results)


# ── Export / Import ───────────────────────────────────────────────────────────

@app.route("/api/export/attendance/csv", methods=["GET"])
def export_att_csv():
    sid = request.args.get("session_id")
    records = get_attendance_for_session(int(sid)) if sid else get_today_attendance()
    sess_rows = get_all_sessions()
    sname = next((s["name"] for s in sess_rows if str(s["id"]) == str(sid)), "today")
    return send_file(export_attendance_csv(records, sname), as_attachment=True)


@app.route("/api/export/attendance/excel", methods=["GET"])
def export_att_excel():
    sid = request.args.get("session_id")
    records = get_attendance_for_session(int(sid)) if sid else get_today_attendance()
    sess_rows = get_all_sessions()
    sname = next((s["name"] for s in sess_rows if str(s["id"]) == str(sid)), "today")
    return send_file(export_attendance_excel(records, sname), as_attachment=True)


@app.route("/api/export/persons/csv", methods=["GET"])
def export_persons():
    return send_file(export_persons_csv(get_all_persons()), as_attachment=True)


@app.route("/api/export/full", methods=["GET"])
def export_full():
    sessions = get_all_sessions()
    all_att  = []
    for s in sessions:
        for r in get_attendance_for_session(s["id"]):
            r["session_name"] = s["name"]
            all_att.append(r)
    return send_file(export_all_sessions_excel(sessions, all_att), as_attachment=True)


@app.route("/api/import/persons", methods=["POST"])
def import_persons():
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "No file"}), 400
    tmp = Path("/tmp") / f.filename
    f.save(str(tmp))
    try:
        rows = import_persons_from_csv(str(tmp))
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    added = skipped = 0
    for row in rows:
        try:
            add_person(str(row.get("name", "")), str(row.get("employee_id", "")),
                       str(row.get("department", "")), str(row.get("email", "")))
            added += 1
        except Exception:
            skipped += 1
    recognizer.load_from_db()
    return jsonify({"added": added, "skipped": skipped})


@app.route("/api/exports/list", methods=["GET"])
def list_export_files():
    return jsonify(list_exports())


@app.route("/api/recognizer/reload", methods=["POST"])
def reload_recognizer():
    recognizer.load_from_db()
    return jsonify({"embeddings": len(recognizer)})


if __name__ == "__main__":
    print("=" * 55)
    print("  AI Attendance System — Backend")
    print("  http://localhost:5000")
    print("=" * 55)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)