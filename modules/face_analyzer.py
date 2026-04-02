"""
CIS v2 - Face Analyzer Module
Extracts facial features using OpenCV Haar cascades.
Returns structured feature dataclass + quality score.
"""
import cv2
import numpy as np
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    FACE_SCALE_FACTOR, FACE_MIN_NEIGHBORS, FACE_MIN_SIZE,
    EYE_SCALE_FACTOR, EYE_MIN_NEIGHBORS
)


class FaceFeatures:
    """Structured container for extracted face features."""
    def __init__(self):
        self.detected         = False
        self.masked           = False          # Face partially hidden
        self.quality          = 0.0            # 0–1 (higher = better sample)
        self.confidence       = 0.0            # 0–1 suspicion score

        # Bounding box
        self.bbox             = None           # (x, y, w, h)

        # Geometric
        self.eye_count        = 0
        self.eye_distance     = 0.0            # Normalized [0–1]
        self.face_area        = 0
        self.face_ratio       = 0.0            # w/h
        self.face_symmetry    = 0.0            # 0-1

        # Intensity
        self.brightness       = 0.0
        self.contrast         = 0.0
        self.sharpness        = 0.0            # Laplacian variance

        # Head pose (approximate)
        self.head_tilt_deg    = 0.0

        # Feature vector for DB comparison (32-dim)
        self.feature_vector   = []

        # Raw
        self.face_roi         = None
        self.gray_roi         = None

    def to_dict(self):
        return {
            "detected":     self.detected,
            "masked":       self.masked,
            "quality":      round(self.quality, 3),
            "confidence":   round(self.confidence, 3),
            "eye_count":    self.eye_count,
            "eye_distance": round(self.eye_distance, 3),
            "face_ratio":   round(self.face_ratio, 3),
            "brightness":   round(self.brightness, 1),
            "contrast":     round(self.contrast, 1),
            "sharpness":    round(self.sharpness, 1),
            "head_tilt":    round(self.head_tilt_deg, 1),
        }


class FaceAnalyzer:
    """
    Multi-stage face analyzer.
    1. Detect face(s) in frame
    2. Extract geometric & intensity features
    3. Detect occlusion (masked face)
    4. Compute 32-dim feature vector
    5. Score suspiciousness
    """

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )
        self.profile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_profileface.xml"
        )
        self._rng = np.random.default_rng(seed=None)

    # ──────────────────────────────────────────────────────────────────────────
    def analyze(self, frame: np.ndarray) -> FaceFeatures:
        """
        Main entry point. Accepts a BGR frame (numpy array).
        Returns a FaceFeatures object.
        """
        ff = FaceFeatures()
        if frame is None or frame.size == 0:
            return ff

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        # ── 1. Detect face ────────────────────────────────────────────────────
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=FACE_SCALE_FACTOR,
            minNeighbors=FACE_MIN_NEIGHBORS,
            minSize=FACE_MIN_SIZE,
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) == 0:
            # Try profile view
            faces = self.profile_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40)
            )

        if len(faces) == 0:
            ff.detected = False
            return ff

        # Choose largest face
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        ff.detected = True
        ff.bbox = (x, y, w, h)

        face_roi  = frame[y:y+h, x:x+w]
        gray_roi  = gray[y:y+h, x:x+w]
        ff.face_roi  = face_roi
        ff.gray_roi  = gray_roi
        ff.face_area = w * h
        ff.face_ratio = round(w / max(h, 1), 3)

        # ── 2. Eye detection ──────────────────────────────────────────────────
        upper_half = gray_roi[:h // 2, :]
        eyes = self.eye_cascade.detectMultiScale(
            upper_half, scaleFactor=EYE_SCALE_FACTOR, minNeighbors=EYE_MIN_NEIGHBORS
        )
        ff.eye_count = len(eyes)

        if len(eyes) >= 2:
            # Normalized inter-ocular distance
            ex1, ey1, ew1, eh1 = eyes[0]
            ex2, ey2, ew2, eh2 = eyes[1]
            cx1 = ex1 + ew1 / 2
            cx2 = ex2 + ew2 / 2
            iod = abs(cx2 - cx1)
            ff.eye_distance = round(iod / max(w, 1), 3)
        elif len(eyes) == 1:
            ff.eye_distance = 0.18
        else:
            ff.masked = True
            ff.eye_distance = 0.0

        # ── 3. Intensity metrics ──────────────────────────────────────────────
        ff.brightness = float(np.mean(gray_roi))
        ff.contrast   = float(np.std(gray_roi))
        ff.sharpness  = float(cv2.Laplacian(gray_roi, cv2.CV_64F).var())

        # ── 4. Head tilt (approx via bounding box diagonal) ───────────────────
        if len(eyes) >= 2:
            ex1, ey1, ew1, _ = eyes[0]
            ex2, ey2, ew2, _ = eyes[1]
            dx = (ex2 + ew2 / 2) - (ex1 + ew1 / 2)
            dy = ey2 - ey1
            ff.head_tilt_deg = round(math.degrees(math.atan2(dy, max(abs(dx), 1))), 1)
        else:
            ff.head_tilt_deg = 0.0

        # ── 5. Face symmetry ──────────────────────────────────────────────────
        left_half  = gray_roi[:, :w // 2]
        right_half = cv2.flip(gray_roi[:, w // 2:], 1)
        min_cols = min(left_half.shape[1], right_half.shape[1])
        if min_cols > 0:
            diff = cv2.absdiff(left_half[:, :min_cols], right_half[:, :min_cols])
            symmetry_err = np.mean(diff) / 255.0
            ff.face_symmetry = round(1.0 - symmetry_err, 3)
        else:
            ff.face_symmetry = 0.5

        # ── 6. Build 32-dim feature vector ────────────────────────────────────
        ff.feature_vector = self._build_feature_vector(ff, gray_roi)

        # ── 7. Compute quality score ──────────────────────────────────────────
        ff.quality = self._compute_quality(ff)

        # ── 8. Compute suspicion confidence ───────────────────────────────────
        ff.confidence = self._compute_suspicion(ff)

        return ff

    # ──────────────────────────────────────────────────────────────────────────
    def _build_feature_vector(self, ff: FaceFeatures, gray_roi: np.ndarray) -> list:
        """Build a 32-dimensional feature vector for DB comparison."""
        # Resize face to 8x4 = 32 pixels, normalize
        try:
            resized = cv2.resize(gray_roi, (8, 4), interpolation=cv2.INTER_AREA)
            pixel_features = (resized.flatten().astype(float) / 255.0).tolist()
        except Exception:
            pixel_features = [0.0] * 32

        # Inject geometric params into last 4 dims
        pixel_features[28] = ff.eye_distance
        pixel_features[29] = ff.face_ratio
        pixel_features[30] = ff.brightness / 255.0
        pixel_features[31] = ff.contrast / 128.0

        return pixel_features

    def _compute_quality(self, ff: FaceFeatures) -> float:
        """Quality = how usable is this sample. Range 0–1."""
        score = 0.5

        # Good face area
        if 4000 < ff.face_area < 80000:
            score += 0.15
        elif ff.face_area < 1600:
            score -= 0.20

        # Sharpness
        if ff.sharpness > 100:
            score += 0.15
        elif ff.sharpness < 30:
            score -= 0.15

        # Eyes visible
        if ff.eye_count >= 2:
            score += 0.20
        elif ff.eye_count == 1:
            score += 0.05

        # Good brightness
        if 75 < ff.brightness < 200:
            score += 0.10

        return float(np.clip(score, 0.0, 1.0))

    def _compute_suspicion(self, ff: FaceFeatures) -> float:
        """
        Compute a suspicion score 0–1 based on face features.
        This is used by the fusion engine as the 'face confidence'.
        """
        score = 0.40  # Neutral baseline (slightly innocent)

        # Masked face → highly suspicious (can't see identity)
        if ff.masked:
            score += 0.22

        # Avoidance: very few eyes detected
        if ff.eye_count == 0:
            score += 0.15
        elif ff.eye_count >= 2:
            score -= 0.08    # Eye contact = less evasive

        # Extreme head tilt
        tilt = abs(ff.head_tilt_deg)
        if tilt > 20:
            score += 0.08
        elif tilt < 5:
            score -= 0.05

        # Abnormal brightness (hiding in shadows or over-exposed)
        if ff.brightness < 60 or ff.brightness > 220:
            score += 0.08

        # Low sharpness (blurry = trying to evade)
        if ff.sharpness < 25:
            score += 0.07

        # Low symmetry (trauma/disguise)
        if ff.face_symmetry < 0.4:
            score += 0.06

        # Add slight randomness for realism (±4%)
        score += self._rng.uniform(-0.04, 0.04)

        return float(np.clip(score, 0.0, 1.0))

    def draw_overlay(self, frame: np.ndarray, ff: FaceFeatures, verdict: str = None,
                     scan_label: str = None) -> np.ndarray:
        """Draw face bounding box + labels on frame. Returns modified copy."""
        out = frame.copy()
        if not ff.detected or ff.bbox is None:
            return out

        x, y, w, h = ff.bbox

        # Color by verdict
        if verdict == "CRIMINAL":
            color = (0, 0, 255)
        elif verdict == "WATCH LIST":
            color = (0, 165, 255)
        elif verdict == "CLEAR":
            color = (0, 200, 60)
        else:
            color = (0, 200, 255)   # Scanning

        # Main bounding box
        cv2.rectangle(out, (x - 8, y - 8), (x + w + 8, y + h + 8), color, 3)

        # Corner accents (tactical UI feel)
        ln = 18
        for px, py, dx, dy in [(x-8, y-8, 1, 1), (x+w+8, y-8, -1, 1),
                                (x-8, y+h+8, 1, -1), (x+w+8, y+h+8, -1, -1)]:
            cv2.line(out, (px, py), (px + dx * ln, py), color, 4)
            cv2.line(out, (px, py), (px, py + dy * ln), color, 4)

        # Header box
        label = scan_label if scan_label else (verdict or "ANALYZING")
        header_y = max(0, y - 45)
        overlay = out.copy()
        cv2.rectangle(overlay, (x - 8, header_y), (x + w + 8, y - 8), color, -1)
        cv2.addWeighted(overlay, 0.7, out, 0.3, 0, out)
        cv2.putText(out, label, (x - 4, y - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # Confidence text below
        conf_text = f"Face: {ff.confidence:.0%}  Q:{ff.quality:.0%}"
        cv2.putText(out, conf_text, (x - 5, y + h + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if ff.masked:
            cv2.putText(out, "⚠ MASKED", (x, y + h + 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        return out
