"""
anti_spoofing.py — v3: SCORE-GATE REJECTION

New approach:
  - Compute a PHOTO SCORE (0–100). If >= 40 → REJECT as photo/spoof.
  - Liveness score = 100 - photo_score (clamped).
  - Attendance only allowed if liveness_score >= 60.

Photo detection signals (each contributes to photo_score):
  1. FLATNESS  — Laplacian variance. Real faces are 3D and sharp.
                 Photos/screens are flat → low Laplacian.
  2. STATIC    — Frame-to-frame pixel diff. Photos don't move at all.
  3. GRADIENT  — Gradient magnitude distribution. Photos have hard compression edges.
  4. DEPTH VAR — Variance of local contrast patches. Flat=photo, varied=real.
  5. BLINK     — No blink detected after N frames → strong photo indicator.
  6. MOIRE     — Screen moire pattern detection via FFT peak analysis.

Rejection rule:  photo_score >= 40  →  REJECTED  (liveness_score < 60)
Acceptance rule: photo_score < 40   →  ACCEPTED   (liveness_score >= 60)
"""

import cv2
import numpy as np
from collections import deque

# ── Thresholds ────────────────────────────────────────────────────────────────

LIVENESS_THRESHOLD   = 60    # liveness_score must be >= this (0–100 scale)
PHOTO_REJECT_SCORE   = 40    # photo_score >= this → reject
WARMUP_FRAMES        = 30    # don't pass before this many frames
MIN_MOTION_MEAN      = 1.2   # mean pixel diff below this = static = photo
BLINK_REQUIRED_AFTER = 60    # frames — if no blink by this frame, penalise hard
LAP_REAL_MIN         = 80.0  # Laplacian variance — real faces typically >80
LAP_PHOTO_MAX        = 50.0  # Photos/screens typically <50


# ── Signal: Laplacian flatness ────────────────────────────────────────────────

def _flatness_score(gray: np.ndarray) -> float:
    """
    Returns 0–100 photo contribution.
    Low Laplacian variance = flat = photo-like.
    Real face: >80  → score ~0
    Photo:     <50  → score ~60–80
    """
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if lap_var >= LAP_REAL_MIN:
        return 0.0
    elif lap_var <= LAP_PHOTO_MAX:
        return 70.0
    else:
        # Linear interpolation between 50 and 80
        ratio = 1.0 - (lap_var - LAP_PHOTO_MAX) / (LAP_REAL_MIN - LAP_PHOTO_MAX)
        return round(ratio * 70.0, 1)


# ── Signal: Static frame (no motion) ─────────────────────────────────────────

def _static_score(motion_history: deque) -> float:
    """
    Returns 0–100 photo contribution.
    Near-zero motion across many frames = photo.
    """
    if len(motion_history) < 8:
        return 0.0
    arr = np.array(motion_history)
    mean_m = float(arr.mean())
    static_frames = int((arr < MIN_MOTION_MEAN).sum())
    static_ratio = static_frames / len(arr)

    if mean_m < MIN_MOTION_MEAN and static_ratio > 0.80:
        return 80.0   # near-perfectly static = photo
    elif mean_m < MIN_MOTION_MEAN * 2 and static_ratio > 0.60:
        return 50.0
    else:
        return 0.0


# ── Signal: Gradient distribution (compression artifacts) ────────────────────

def _gradient_score(gray: np.ndarray) -> float:
    """
    JPEG/screen photos have characteristic gradient histograms.
    Real faces: smooth gradient distribution.
    Photos: spiky histogram (compression quantisation).
    """
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2).astype(np.uint8)
    hist, _ = np.histogram(mag.ravel(), bins=32, range=(0, 256))
    hist = hist.astype(float) / (hist.sum() + 1e-9)
    # Measure peakiness: high kurtosis = photo compression artefacts
    mean = np.average(np.arange(32), weights=hist)
    var  = np.average((np.arange(32) - mean)**2, weights=hist)
    if var < 1e-6:
        return 40.0
    kurt = np.average((np.arange(32) - mean)**4, weights=hist) / (var**2 + 1e-9)
    # Real face: kurt ~2–6. Photo: kurt >8 (spiky)
    if kurt > 10:
        return 45.0
    elif kurt > 7:
        return 25.0
    else:
        return 0.0


# ── Signal: Local depth variance ─────────────────────────────────────────────

def _depth_variance_score(gray: np.ndarray) -> float:
    """
    Divide face into 4x4 patches. Compute variance of each patch's mean intensity.
    Real faces: high spatial variance (nose ridge, eye sockets, cheeks differ).
    Photos held at arm's length: more uniform.
    """
    h, w = gray.shape
    ph, pw = h // 4, w // 4
    means = []
    for r in range(4):
        for c in range(4):
            patch = gray[r*ph:(r+1)*ph, c*pw:(c+1)*pw]
            means.append(float(patch.mean()))
    spatial_var = float(np.var(means))
    # Real: >150. Photo: <80 (flat lighting on flat surface)
    if spatial_var >= 150:
        return 0.0
    elif spatial_var <= 60:
        return 50.0
    else:
        ratio = 1.0 - (spatial_var - 60) / 90.0
        return round(ratio * 50.0, 1)


# ── Signal: Moire/screen pattern (FFT peak detection) ────────────────────────

def _moire_score(gray: np.ndarray) -> float:
    """
    Screens showing a photo create moire patterns detectable in FFT.
    Looks for unnatural periodic peaks away from DC component.
    """
    f = np.abs(np.fft.fftshift(np.fft.fft2(gray.astype(np.float32))))
    h, w = f.shape
    cy, cx = h // 2, w // 2
    # Blank DC component
    f[cy-3:cy+3, cx-3:cx+3] = 0
    # Normalise
    f = f / (f.max() + 1e-9)
    # Find strong off-centre peaks
    threshold = 0.35
    peak_mask = (f > threshold).astype(np.uint8)
    peak_count = int(peak_mask.sum())
    # Real faces: 0–3 peaks. Screens: many symmetric peaks
    if peak_count >= 8:
        return 55.0
    elif peak_count >= 4:
        return 25.0
    else:
        return 0.0


# ── Blink detector ────────────────────────────────────────────────────────────

_eye_cascade = None

def _get_eye_cascade():
    global _eye_cascade
    if _eye_cascade is None:
        _eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    return _eye_cascade


def _eye_openness(gray_face: np.ndarray) -> float:
    eyes = _get_eye_cascade().detectMultiScale(gray_face, 1.1, 4, minSize=(12, 12))
    if len(eyes) == 0:
        return -1.0
    face_area = gray_face.shape[0] * gray_face.shape[1]
    total = sum(ew * eh for (_, _, ew, eh) in eyes[:2])
    return min(total / (face_area * 0.15 + 1e-6), 1.0)


# ── Main class ────────────────────────────────────────────────────────────────

class LivenessChecker:
    """
    Computes a photo_score (0–100).
    If photo_score >= PHOTO_REJECT_SCORE → REJECTED.
    liveness_score = max(0, 100 - photo_score).
    Attendance allowed only if liveness_score >= LIVENESS_THRESHOLD (60).
    """

    def __init__(self):
        self.frame_count       = 0
        self.motion_history    = deque(maxlen=25)
        self.score_history     = deque(maxlen=10)
        self.prev_gray         = None

        # Blink tracking
        self.blink_count       = 0
        self.eye_baseline      = None
        self.eye_history       = deque(maxlen=8)
        self._blink_state      = "OPEN"

        self.last_cues         = {}

    # ── Motion ────────────────────────────────────────────────────────────────

    def _update_motion(self, gray: np.ndarray) -> float:
        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray.copy()
            return 0.0
        diff = cv2.absdiff(gray, self.prev_gray)
        m = float(diff.mean())
        self.prev_gray = gray.copy()
        self.motion_history.append(m)
        return m

    # ── Blink state machine ───────────────────────────────────────────────────

    def _update_blink(self, openness: float):
        if openness < 0:
            return
        self.eye_history.append(openness)
        if self.eye_baseline is None and len(self.eye_history) >= 5:
            candidates = [v for v in self.eye_history if v > 0.15]
            if candidates:
                self.eye_baseline = float(np.mean(candidates))
        if self.eye_baseline is None:
            return
        rel = openness / (self.eye_baseline + 1e-6)
        if self._blink_state == "OPEN" and rel < 0.55:
            self._blink_state = "CLOSING"
        elif self._blink_state == "CLOSING":
            if rel < 0.35:
                self._blink_state = "CLOSED"
            elif rel > 0.70:
                self._blink_state = "OPEN"
        elif self._blink_state == "CLOSED" and rel > 0.55:
            self._blink_state = "OPENING"
        elif self._blink_state == "OPENING" and rel > 0.75:
            self._blink_state = "OPEN"
            self.blink_count += 1

    # ── Main update ───────────────────────────────────────────────────────────

    def update(self, frame: np.ndarray, face_box: tuple) -> dict:
        self.frame_count += 1
        x, y, w, h = [int(v) for v in face_box]
        fh, fw = frame.shape[:2]
        x, y = max(0, x), max(0, y)
        w = min(w, fw - x)
        h = min(h, fh - y)

        face_bgr = frame[y:y+h, x:x+w]
        if face_bgr.size == 0:
            return {"liveness_score": 0, "photo_score": 100, "is_live": False}

        face128 = cv2.resize(face_bgr, (128, 128))
        gray    = cv2.cvtColor(face128, cv2.COLOR_BGR2GRAY)

        # Update motion
        self._update_motion(gray)

        # Update blink
        openness = _eye_openness(gray)
        self._update_blink(openness)

        # ── Compute photo score ────────────────────────────────────────────

        s_flat    = _flatness_score(gray)
        s_static  = _static_score(self.motion_history)
        s_grad    = _gradient_score(gray)
        s_depth   = _depth_variance_score(gray)
        s_moire   = _moire_score(gray)

        # Blink penalty: no blink after threshold frames = photo-like
        blink_penalty = 0.0
        if self.frame_count > BLINK_REQUIRED_AFTER and self.blink_count == 0:
            blink_penalty = 35.0
        elif self.frame_count > BLINK_REQUIRED_AFTER * 2 and self.blink_count == 0:
            blink_penalty = 55.0

        # Weighted combination into photo_score
        # Static and flatness are the strongest individual signals
        photo_score_raw = (
            s_flat   * 0.30 +
            s_static * 0.30 +
            s_grad   * 0.10 +
            s_depth  * 0.15 +
            s_moire  * 0.05 +
            blink_penalty * 0.10
        )

        # Hard override: if BOTH flat AND static → definitely a photo
        if s_flat >= 60 and s_static >= 60:
            photo_score_raw = max(photo_score_raw, 75.0)

        # Hard override: if static alone is very high (nearly zero motion)
        if s_static >= 80:
            photo_score_raw = max(photo_score_raw, 65.0)

        photo_score = min(round(photo_score_raw, 1), 100.0)

        # Warmup: during warmup, defer judgment (neutral score)
        if self.frame_count < WARMUP_FRAMES:
            # During warmup we still track but don't finalize
            liveness_score = 50  # neutral — don't pass or block yet
        else:
            liveness_score = max(0, round(100.0 - photo_score, 1))

        # Smooth liveness score
        self.score_history.append(liveness_score)
        smoothed = round(float(np.mean(list(self.score_history))), 1)

        is_live = smoothed >= LIVENESS_THRESHOLD and self.frame_count >= WARMUP_FRAMES

        self.last_cues = {
            "liveness_score":  smoothed,
            "photo_score":     photo_score,
            "is_live":         is_live,
            "signals": {
                "flatness":    round(s_flat, 1),
                "static":      round(s_static, 1),
                "gradient":    round(s_grad, 1),
                "depth_var":   round(s_depth, 1),
                "moire":       round(s_moire, 1),
                "blink_pen":   round(blink_penalty, 1),
            },
            "blink_count":  self.blink_count,
            "frame":        self.frame_count,
        }
        return self.last_cues

    def reset(self):
        self.frame_count    = 0
        self.motion_history.clear()
        self.score_history.clear()
        self.prev_gray      = None
        self.blink_count    = 0
        self.eye_baseline   = None
        self.eye_history.clear()
        self._blink_state   = "OPEN"
        self.last_cues      = {}

    @property
    def is_live(self) -> bool:
        if not self.score_history:
            return False
        return float(np.mean(list(self.score_history))) >= LIVENESS_THRESHOLD and \
               self.frame_count >= WARMUP_FRAMES

    def get_status_message(self) -> str:
        cues = self.last_cues
        if not cues:
            return "Initialising..."
        if self.frame_count < WARMUP_FRAMES:
            remaining = WARMUP_FRAMES - self.frame_count
            return f"Please wait... ({remaining} frames)"
        sig = cues.get("signals", {})
        ps  = cues.get("photo_score", 100)
        if sig.get("static", 0) >= 60:
            return "⛔ PHOTO DETECTED — Please show your real face"
        if sig.get("flatness", 0) >= 60:
            return "⛔ FLAT IMAGE — Move closer, check lighting"
        if sig.get("moire", 0) >= 40:
            return "⛔ SCREEN DETECTED — No screen/device photos"
        if cues.get("blink_count", 0) == 0 and self.frame_count > BLINK_REQUIRED_AFTER:
            return "👁 Please blink naturally"
        if ps >= PHOTO_REJECT_SCORE:
            return f"⛔ Rejected (photo score {ps:.0f}/100)"
        return f"✓ Live ({100 - ps:.0f}/100)"
