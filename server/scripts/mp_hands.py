"""Shared MediaPipe hand-landmark utilities.

Wraps the MediaPipe Tasks HandLandmarker (works on mediapipe >=0.10 and 1.x).
Auto-downloads the .task model file on first use.

Landmark index reference (MediaPipe hand model):
    0 WRIST
    5 INDEX_MCP   6 INDEX_PIP   7 INDEX_DIP   8 INDEX_TIP
    9 MIDDLE_MCP 10 MIDDLE_PIP 11 MIDDLE_DIP 12 MIDDLE_TIP
   13 RING_MCP   14 RING_PIP   15 RING_DIP   16 RING_TIP
   17 PINKY_MCP  18 PINKY_PIP  19 PINKY_DIP  20 PINKY_TIP
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "models"
MODEL_PATH = MODEL_DIR / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGERS = {
    "index": (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    "pinky": (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def ensure_model() -> Path:
    if not MODEL_PATH.exists():
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading hand_landmarker model to {MODEL_PATH} ...")
        tmp = MODEL_PATH.with_suffix(".tmp")
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.replace(MODEL_PATH)
    return MODEL_PATH


def imread_unicode(path: str | Path) -> np.ndarray | None:
    """cv2.imread fails on non-ASCII Windows paths; decode from bytes instead."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


class HandDetector:
    """Single-image hand landmark detector returning pixel coordinates."""

    def __init__(self, num_hands: int = 1, min_detection_confidence: float = 0.5):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model())),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
        )
        self._detector = vision.HandLandmarker.create_from_options(options)

    def detect(self, image: str | Path | np.ndarray) -> dict | None:
        """Detect the most confident hand.

        Returns None if no hand is found, else a dict with:
            landmarks_px : (21, 2) float array, pixel coords
            landmarks_z  : (21,) relative depth (wrist-origin, MediaPipe units)
            handedness   : "Left" or "Right" (anatomical hand, as predicted)
            score        : handedness confidence
            width, height: image dimensions
        """
        if isinstance(image, (str, Path)):
            bgr = imread_unicode(image)
            if bgr is None:
                return None
        else:
            bgr = image
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)
        del mp_image, rgb
        if not result.hand_landmarks:
            return None
        lms = result.hand_landmarks[0]
        cat = result.handedness[0][0]
        px = np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=float)
        z = np.array([lm.z for lm in lms], dtype=float)
        return {
            "landmarks_px": px,
            "landmarks_z": z,
            "handedness": cat.category_name,
            "score": float(cat.score),
            "width": w,
            "height": h,
        }

    def close(self):
        self._detector.close()


def hand_size_px(landmarks_px: np.ndarray) -> float:
    """Palm length: wrist to middle-finger MCP. Standard normaliser for offsets."""
    return float(np.linalg.norm(landmarks_px[MIDDLE_MCP] - landmarks_px[WRIST]))


def crease_estimate(landmarks_px: np.ndarray, mcp: int, pip: int, offset_frac: float) -> np.ndarray:
    """Estimate the basal palmar crease from the MCP landmark.

    MediaPipe's MCP landmarks (5/13) sit on the metacarpal head, not the crease.
    We shift along the finger axis (MCP -> PIP) by offset_frac of the
    MCP-PIP segment length. Positive = distal (toward the fingertip),
    negative = proximal (toward the wrist). Axis-based rather than image-"down"
    so it is invariant to hand orientation in the frame.
    """
    return landmarks_px[mcp] + offset_frac * (landmarks_px[pip] - landmarks_px[mcp])
