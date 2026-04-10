"""
registration.py
Enroll a new person by capturing multiple frames from a live webcam feed,
extracting FaceNet embeddings, averaging them, and storing in the database.
"""

import cv2
import numpy as np
import logging
import os
from typing import Optional, List
import time

from face_engine import (detect_faces, align_and_crop,
                         get_embedding, average_embedding)
from anti_spoof import LivenessChecker
import database_manager as db

logger = logging.getLogger(__name__)

CAPTURE_SAMPLES = 15          # frames to capture per person
SAMPLE_INTERVAL_SEC = 0.25    # seconds between captures
FACE_IMAGES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "registered_faces"
)
os.makedirs(FACE_IMAGES_DIR, exist_ok=True)


def enroll_person(name: str, employee_id: str,
                  department: str = "",
                  email: str = "",
                  phone: str = "",
                  camera_id: int = 0,
                  samples: int = CAPTURE_SAMPLES) -> Optional[int]:
    """
    Open webcam, collect `samples` face images, compute mean embedding,
    and store in DB.  Returns DB person_id or None on failure.
    """
    # Check for duplicate employee_id
    if db.get_person_by_employee_id(employee_id):
        logger.error(f"Employee ID {employee_id!r} already exists.")
        return None

    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
    if not cap.isOpened():
        logger.error("Cannot open camera.")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    liveness = LivenessChecker()
    embeddings: List[np.ndarray] = []
    thumbnail: Optional[np.ndarray] = None
    collected = 0
    last_capture = 0.0

    print(f"\n[ENROLL] Registering: {name} ({employee_id})")
    print(f"[ENROLL] Look at the camera. Collecting {samples} samples…\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        faces = detect_faces(frame)

        for face in faces:
            if face["confidence"] < 0.97:
                continue
            box = face["box"]
            kpts = face["keypoints"]

            face_crop = align_and_crop(frame, box, kpts)
            if face_crop is None:
                continue

            # Liveness
            is_live, reason, _ = liveness.check(
                face_crop, face_box=box, frame_shape=frame.shape, keypoints=kpts
            )

            x, y, w, h = box
            color = (0, 200, 0) if is_live else (0, 0, 200)
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)

            if is_live and (time.time() - last_capture) > SAMPLE_INTERVAL_SEC:
                emb = get_embedding(face_crop)
                embeddings.append(emb)
                collected += 1
                last_capture = time.time()
                if thumbnail is None:
                    thumbnail = face_crop.copy()
                print(f"  Sample {collected}/{samples}", end="\r", flush=True)

            status = f"{collected}/{samples} | {'LIVE' if is_live else 'SPOOF'}"
            cv2.putText(display, status, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(display, f"Enrolling: {name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.imshow("Enrollment - Press Q to cancel", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n[ENROLL] Cancelled.")
            break

        if collected >= samples:
            print(f"\n[ENROLL] Captured {collected} samples.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(embeddings) < 5:
        logger.error(f"Not enough valid samples ({len(embeddings)}). Enrolment failed.")
        return None

    mean_emb = average_embedding(embeddings)

    # Save thumbnail
    thumb_bytes = None
    if thumbnail is not None:
        thumb_path = os.path.join(FACE_IMAGES_DIR, f"{employee_id}.jpg")
        cv2.imwrite(thumb_path, thumbnail)
        _, buf = cv2.imencode(".jpg", thumbnail)
        thumb_bytes = buf.tobytes()

    person_id = db.add_person(
        name=name,
        employee_id=employee_id,
        embedding=mean_emb,
        department=department,
        email=email,
        phone=phone,
        face_image=thumb_bytes,
    )
    print(f"[ENROLL] ✓ {name} registered (id={person_id}).")
    return person_id


def re_enroll_person(employee_id: str,
                     camera_id: int = 0,
                     samples: int = CAPTURE_SAMPLES) -> bool:
    """Update embedding for an existing person."""
    person = db.get_person_by_employee_id(employee_id)
    if not person:
        logger.error(f"Employee {employee_id!r} not found.")
        return False

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        return False

    liveness = LivenessChecker()
    embeddings: List[np.ndarray] = []
    collected = 0
    last_capture = 0.0

    print(f"[RE-ENROLL] Updating {person['name']}…")

    while collected < samples:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_faces(frame)
        for face in faces:
            if face["confidence"] < 0.97:
                continue
            face_crop = align_and_crop(frame, face["box"], face["keypoints"])
            if face_crop is None:
                continue
            is_live, _, _ = liveness.check(face_crop)
            if is_live and (time.time() - last_capture) > SAMPLE_INTERVAL_SEC:
                embeddings.append(get_embedding(face_crop))
                collected += 1
                last_capture = time.time()

        cv2.imshow("Re-enrollment", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(embeddings) < 5:
        return False

    db.update_person_embedding(person["id"], average_embedding(embeddings))
    print(f"[RE-ENROLL] ✓ Embedding updated for {person['name']}.")
    return True
