"""
anti_spoof.py
Multi-method passive liveness / anti-spoofing for the attendance system.

Methods used (no IR camera needed):
  1. Laplacian blur detection  – printed photos are usually blurry at edges
  2. LBP texture analysis      – screen / paper lack natural skin micro-texture
  3. Frequency domain analysis – replay / print artefacts show as periodic noise
  4. Eye-blink detection       – optional, requires landmark sequence over frames
  5. Face ratio sanity check   – face must be a reasonable fraction of frame
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


# ── Individual Detectors ──────────────────────────────────────────────────────

def check_blur(face_bgr: np.ndarray, threshold: float = 80.0) -> Tuple[bool, float]:
    """
    Laplacian variance test.
    Live faces have higher-frequency detail than photos/screens.
    Returns (is_live, variance).
    """
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_live = lap_var >= threshold
    return is_live, float(lap_var)


def check_lbp_texture(face_bgr: np.ndarray,
                      threshold: float = 12.0) -> Tuple[bool, float]:
    """
    Local Binary Pattern entropy.
    Printed / screen images have lower texture entropy than real skin.
    """
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64))

    # Compute LBP manually (radius=1, 8 neighbours)
    rows, cols = gray.shape
    lbp = np.zeros_like(gray, dtype=np.uint8)
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            center = gray[i, j]
            code = 0
            for k, (di, dj) in enumerate([(-1,-1),(-1,0),(-1,1),
                                           (0,1),(1,1),(1,0),(1,-1),(0,-1)]):
                if gray[i+di, j+dj] >= center:
                    code |= (1 << k)
            lbp[i, j] = code

    # Histogram entropy
    hist = np.bincount(lbp.ravel(), minlength=256).astype(np.float32)
    hist /= (hist.sum() + 1e-9)
    entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
    is_live = entropy >= threshold
    return is_live, float(entropy)


def check_frequency_domain(face_bgr: np.ndarray,
                            threshold: float = 0.35) -> Tuple[bool, float]:
    """
    High-frequency content ratio via FFT.
    Printed / replayed faces attenuate high-freq components.
    """
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.resize(gray, (128, 128))

    dft = np.fft.fft2(gray)
    dft_shift = np.fft.fftshift(dft)
    magnitude = np.abs(dft_shift)

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    # Inner (low-freq) region radius = 20% of min dimension
    r = int(min(h, w) * 0.20)
    Y, X = np.ogrid[:h, :w]
    mask_low = (X - cx)**2 + (Y - cy)**2 <= r**2

    total_power = magnitude.sum() + 1e-9
    high_power  = magnitude[~mask_low].sum()
    hf_ratio    = float(high_power / total_power)

    is_live = hf_ratio >= threshold
    return is_live, hf_ratio


def check_skin_color(face_bgr: np.ndarray,
                     min_ratio: float = 0.20) -> Tuple[bool, float]:
    """
    Check that the face region contains enough skin-tone pixels.
    Extreme mismatch can indicate a printed image or spoofing artefact.
    """
    ycrcb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2YCrCb)
    # Skin tone range in YCrCb
    lower = np.array([0,  133, 77], dtype=np.uint8)
    upper = np.array([235, 173, 127], dtype=np.uint8)
    mask  = cv2.inRange(ycrcb, lower, upper)
    ratio = float(mask.sum()) / (mask.size * 255 + 1e-9)
    return ratio >= min_ratio, ratio


def check_face_size_ratio(face_box: list, frame_shape: tuple,
                           min_ratio: float = 0.04,
                           max_ratio: float = 0.90) -> Tuple[bool, float]:
    """
    Face must occupy a reasonable fraction of the frame.
    Tiny boxes are distant faces; huge boxes are likely photos held very close.
    """
    x, y, w, h = face_box
    fh, fw = frame_shape[:2]
    face_ratio = (w * h) / (fw * fh + 1e-9)
    is_ok = min_ratio <= face_ratio <= max_ratio
    return is_ok, float(face_ratio)


# ── Blink Tracker ─────────────────────────────────────────────────────────────

class BlinkTracker:
    """
    Tracks Eye Aspect Ratio (EAR) across frames to detect blinks.
    Uses MTCNN keypoints (left_eye, right_eye) — no dlib needed.
    A blink is detected when EAR drops below threshold and rises again.
    """
    EAR_THRESHOLD = 0.22
    CONSEC_FRAMES = 2

    def __init__(self):
        self._counter = 0
        self._total   = 0
        self._prev_ear = 1.0

    def update(self, keypoints: Optional[dict]) -> int:
        """Feed latest MTCNN keypoints; returns cumulative blink count."""
        if keypoints is None:
            return self._total
        # MTCNN gives only eye centre points, not 6-point contour.
        # Estimate EAR from eye-to-nose distance ratio as a proxy.
        le = np.array(keypoints.get("left_eye",  [0, 0]))
        re = np.array(keypoints.get("right_eye", [0, 0]))
        n  = np.array(keypoints.get("nose",      [0, 0]))

        eye_dist  = np.linalg.norm(le - re) + 1e-9
        eye_nose  = (np.linalg.norm(le - n) + np.linalg.norm(re - n)) / 2
        ear       = float(eye_nose / eye_dist)

        if ear < self.EAR_THRESHOLD:
            self._counter += 1
        else:
            if self._counter >= self.CONSEC_FRAMES:
                self._total += 1
            self._counter = 0
        self._prev_ear = ear
        return self._total

    def reset(self):
        self._counter = 0
        self._total   = 0


# ── Composite Liveness Check ──────────────────────────────────────────────────

class LivenessChecker:
    """
    Aggregates multiple checks with weighted voting.
    At least 3 of 4 static checks must pass for a face to be considered live.
    """

    def __init__(self, blur_thresh=80.0, lbp_thresh=12.0,
                 freq_thresh=0.35, skin_thresh=0.20):
        self.blur_thresh = blur_thresh
        self.lbp_thresh  = lbp_thresh
        self.freq_thresh = freq_thresh
        self.skin_thresh = skin_thresh

    def check(self, face_bgr: np.ndarray,
              face_box: list = None,
              frame_shape: tuple = None,
              keypoints: dict = None) -> Tuple[bool, str, dict]:
        """
        Returns (is_live, reason_string, detail_dict).
        """
        details = {}

        live_blur, v_blur = check_blur(face_bgr, self.blur_thresh)
        details["blur_var"] = round(v_blur, 2)
        details["blur_ok"]  = live_blur

        live_lbp, v_lbp = check_lbp_texture(face_bgr, self.lbp_thresh)
        details["lbp_entropy"] = round(v_lbp, 3)
        details["lbp_ok"]      = live_lbp

        live_freq, v_freq = check_frequency_domain(face_bgr, self.freq_thresh)
        details["freq_ratio"] = round(v_freq, 3)
        details["freq_ok"]    = live_freq

        live_skin, v_skin = check_skin_color(face_bgr, self.skin_thresh)
        details["skin_ratio"] = round(v_skin, 3)
        details["skin_ok"]    = live_skin

        checks = [live_blur, live_lbp, live_freq, live_skin]
        passed = sum(checks)

        if face_box and frame_shape:
            size_ok, face_ratio = check_face_size_ratio(face_box, frame_shape)
            details["face_ratio"] = round(face_ratio, 3)
            details["size_ok"]    = size_ok
            if not size_ok:
                return False, "Face size out of acceptable range", details

        is_live = passed >= 3

        if not is_live:
            failed = []
            if not live_blur: failed.append(f"blur({v_blur:.1f}<{self.blur_thresh})")
            if not live_lbp:  failed.append(f"texture({v_lbp:.2f}<{self.lbp_thresh})")
            if not live_freq: failed.append(f"freq({v_freq:.2f}<{self.freq_thresh})")
            if not live_skin: failed.append(f"skin({v_skin:.2f}<{self.skin_thresh})")
            reason = "Spoof detected: " + ", ".join(failed)
        else:
            reason = "Live"

        return is_live, reason, details
