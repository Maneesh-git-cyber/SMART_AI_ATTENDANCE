"""
face_engine.py — Face Detection (MTCNN) + Recognition (FaceNet/InceptionResnetV1)

Supports up to 150+ registered persons.
Uses cosine similarity with configurable threshold.
"""

import cv2
import numpy as np
import torch
from PIL import Image
from pathlib import Path
import threading

# Lazy imports so the module loads even without GPU
_mtcnn = None
_facenet = None
_device = None
_lock = threading.Lock()


def _get_device():
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def _get_mtcnn():
    global _mtcnn
    if _mtcnn is None:
        from facenet_pytorch import MTCNN
        _mtcnn = MTCNN(
            image_size=160,
            margin=20,
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            keep_all=True,
            device=_get_device(),
        )
    return _mtcnn


def _get_facenet():
    global _facenet
    if _facenet is None:
        from facenet_pytorch import InceptionResnetV1
        _facenet = InceptionResnetV1(pretrained="vggface2").eval().to(_get_device())
    return _facenet


# ── Embedding extraction ──────────────────────────────────────────────────────

def extract_embedding(face_tensor: torch.Tensor) -> np.ndarray:
    """Given a (1,3,160,160) normalised tensor, return 512-d embedding."""
    with torch.no_grad():
        emb = _get_facenet()(face_tensor.to(_get_device()))
    return emb.cpu().numpy().flatten()


def extract_embedding_from_image(img_bgr: np.ndarray) -> np.ndarray | None:
    """Full pipeline: BGR image → embedding (or None if no face)."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    mtcnn = _get_mtcnn()
    with _lock:
        faces = mtcnn(pil_img)  # returns tensor(s) or None
    if faces is None:
        return None
    if faces.ndim == 3:
        faces = faces.unsqueeze(0)
    return extract_embedding(faces[0:1])


# ── Detection ─────────────────────────────────────────────────────────────────

def detect_faces(frame_bgr: np.ndarray):
    """
    Returns list of dicts:
      { 'box': (x,y,w,h), 'prob': float, 'face_tensor': Tensor(1,3,160,160) }
    """
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    mtcnn = _get_mtcnn()

    with _lock:
        boxes, probs, _ = mtcnn.detect(pil_img, landmarks=True)
        face_tensors = mtcnn(pil_img)

    if boxes is None or len(boxes) == 0:
        return []

    results = []
    for i, (box, prob) in enumerate(zip(boxes, probs)):
        if prob is None or prob < 0.90:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        x, y = max(0, x1), max(0, y1)
        w, h = x2 - x1, y2 - y1

        tensor = None
        if face_tensors is not None:
            if face_tensors.ndim == 3:
                tensor = face_tensors.unsqueeze(0)
            else:
                if i < len(face_tensors):
                    tensor = face_tensors[i:i+1]

        results.append({
            "box": (x, y, w, h),
            "prob": float(prob),
            "face_tensor": tensor,
        })
    return results


# ── Recognition ───────────────────────────────────────────────────────────────

class FaceRecognizer:
    """
    In-memory index of known embeddings.
    Thread-safe. Refreshed from DB on demand.
    """

    THRESHOLD = 0.68       # cosine similarity — tune 0.60–0.75
    MIN_EMBEDDINGS = 3     # skip persons with too few samples

    def __init__(self):
        self._embeddings: list[tuple[int, np.ndarray]] = []   # (person_id, emb)
        self._lock = threading.RLock()

    def load_from_db(self):
        """Reload all embeddings from the database."""
        from core.database import load_all_embeddings
        data = load_all_embeddings()
        with self._lock:
            self._embeddings = data
        print(f"[Recognizer] Loaded {len(data)} embeddings.")

    def add_embedding(self, person_id: int, embedding: np.ndarray):
        with self._lock:
            self._embeddings.append((person_id, embedding))

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        a_n = a / (np.linalg.norm(a) + 1e-8)
        b_n = b / (np.linalg.norm(b) + 1e-8)
        return float(np.dot(a_n, b_n))

    def identify(self, query_emb: np.ndarray) -> tuple[int | None, float]:
        """
        Returns (person_id, confidence) or (None, 0.0) if unknown.
        Uses nearest-neighbour with majority-vote across all stored embeddings.
        """
        with self._lock:
            if not self._embeddings:
                return None, 0.0

            # Compute similarities
            sims = [(pid, self._cosine_similarity(query_emb, emb))
                    for pid, emb in self._embeddings]

            # Group by person, take max similarity per person
            from collections import defaultdict
            best_per_person: dict[int, float] = defaultdict(float)
            for pid, sim in sims:
                if sim > best_per_person[pid]:
                    best_per_person[pid] = sim

            if not best_per_person:
                return None, 0.0

            best_pid = max(best_per_person, key=lambda k: best_per_person[k])
            best_sim = best_per_person[best_pid]

            if best_sim < self.THRESHOLD:
                return None, best_sim

            return best_pid, best_sim

    def __len__(self):
        return len(self._embeddings)


# ── Drawing utilities ─────────────────────────────────────────────────────────

def draw_face_box(
    frame: np.ndarray,
    box: tuple,
    label: str,
    confidence: float,
    liveness_score: float,
    color: tuple = (0, 255, 0),
) -> np.ndarray:
    x, y, w, h = box
    # Box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Background for text
    text = f"{label} ({confidence:.0%})"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x, y - th - 10), (x + tw + 6, y), color, -1)
    cv2.putText(frame, text, (x + 3, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    # Liveness bar
    bar_w = w
    bar_h = 5
    bar_y = y + h + 4
    cv2.rectangle(frame, (x, bar_y), (x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
    live_w = int(bar_w * liveness_score)
    live_color = (0, 200, 0) if liveness_score >= 0.45 else (0, 0, 220)
    cv2.rectangle(frame, (x, bar_y), (x + live_w, bar_y + bar_h), live_color, -1)

    return frame
