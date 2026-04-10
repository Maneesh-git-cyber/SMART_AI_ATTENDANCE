"""
face_engine.py
Face detection (MTCNN) + embedding extraction (FaceNet/InceptionResNetV1).
Supports up to 150 registered persons with cosine-similarity matching.
"""

import os
import cv2
import numpy as np
from typing import Optional, List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

# ── Lazy imports so the app starts even if GPU libs are loading ──────────────
_mtcnn = None
_facenet = None


def _load_mtcnn():
    global _mtcnn
    if _mtcnn is None:
        try:
            from mtcnn import MTCNN
            _mtcnn = MTCNN(min_face_size=40, steps_threshold=[0.6, 0.7, 0.9])
            logger.info("MTCNN loaded.")
        except Exception as e:
            logger.error(f"MTCNN load failed: {e}")
            raise
    return _mtcnn


def _load_facenet():
    global _facenet
    if _facenet is None:
        try:
            # keras-facenet wraps the pre-trained InceptionResNetV1
            from keras_facenet import FaceNet
            _facenet = FaceNet()
            logger.info("FaceNet loaded.")
        except Exception as e:
            logger.error(f"FaceNet load failed: {e}")
            raise
    return _facenet


# ── Face Detection ────────────────────────────────────────────────────────────

def detect_faces(frame_bgr: np.ndarray) -> List[Dict]:
    """
    Detect faces in a BGR frame.
    Returns list of dicts with keys: box, confidence, keypoints.
    box = [x, y, w, h]
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    detector = _load_mtcnn()
    results = detector.detect_faces(rgb)
    return results  # list of {'box', 'confidence', 'keypoints'}


def align_and_crop(frame_bgr: np.ndarray, box: List[int],
                   keypoints: Dict, target_size: int = 160) -> np.ndarray:
    """
    Align face using eye keypoints and return a target_size x target_size BGR crop.
    """
    x, y, w, h = box
    # Expand box by 20% for context
    margin_x = int(w * 0.20)
    margin_y = int(h * 0.20)
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(frame_bgr.shape[1], x + w + margin_x)
    y2 = min(frame_bgr.shape[0], y + h + margin_y)

    face_crop = frame_bgr[y1:y2, x1:x2]
    if face_crop.size == 0:
        return None

    # Geometric alignment using eyes
    left_eye  = np.array(keypoints["left_eye"])
    right_eye = np.array(keypoints["right_eye"])

    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))

    center = (frame_bgr.shape[1] // 2, frame_bgr.shape[0] // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned = cv2.warpAffine(frame_bgr, M, (frame_bgr.shape[1], frame_bgr.shape[0]),
                              flags=cv2.INTER_CUBIC)

    # Re-crop after alignment
    face_aligned = aligned[y1:y2, x1:x2]
    if face_aligned.size == 0:
        face_aligned = face_crop

    resized = cv2.resize(face_aligned, (target_size, target_size),
                         interpolation=cv2.INTER_AREA)
    return resized


# ── Embedding Extraction ──────────────────────────────────────────────────────

def get_embedding(face_bgr: np.ndarray) -> np.ndarray:
    """
    Extract a 512-d L2-normalised FaceNet embedding from a BGR face crop.
    """
    model = _load_facenet()
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_rgb = cv2.resize(face_rgb, (160, 160))
    # keras-facenet expects uint8 RGB numpy arrays
    embeddings = model.embeddings([face_rgb])   # shape (1, 512)
    emb = embeddings[0].astype(np.float32)
    # L2 normalise
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb /= norm
    return emb


def average_embedding(embeddings: List[np.ndarray]) -> np.ndarray:
    """Return L2-normalised mean of a list of embeddings."""
    stack = np.stack(embeddings, axis=0)
    mean_emb = stack.mean(axis=0).astype(np.float32)
    norm = np.linalg.norm(mean_emb)
    if norm > 0:
        mean_emb /= norm
    return mean_emb


# ── Matching ─────────────────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [-1, 1]. Higher = more similar."""
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))


def find_best_match(probe_embedding: np.ndarray,
                    gallery: List[Dict],
                    threshold: float = 0.68) -> Tuple[Optional[Dict], float]:
    """
    Compare probe against all gallery entries.
    gallery items must have an 'embedding' key (np.ndarray).
    Returns (best_person_dict or None, similarity_score).
    """
    if not gallery:
        return None, 0.0

    best_score = -1.0
    best_person = None
    for person in gallery:
        score = cosine_similarity(probe_embedding, person["embedding"])
        if score > best_score:
            best_score = score
            best_person = person

    if best_score >= threshold:
        return best_person, best_score
    return None, best_score


# ── Utility ───────────────────────────────────────────────────────────────────

def draw_face_box(frame: np.ndarray, box: List[int], name: str,
                  confidence: float, color: Tuple = (0, 220, 0),
                  spoof: bool = False):
    """Draw bounding box + label on frame (in-place)."""
    x, y, w, h = box
    label_color = (0, 0, 220) if spoof else color

    # Box
    cv2.rectangle(frame, (x, y), (x + w, y + h), label_color, 2)

    # Background for label
    label = f"{'SPOOF!' if spoof else name}  {confidence:.0%}"
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x, y - lh - 8), (x + lw + 6, y), label_color, -1)

    cv2.putText(frame, label, (x + 3, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
