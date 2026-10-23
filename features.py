"""Shared MediaPipe landmark features for collection, training and inference."""
import numpy as np

LANDMARKS_PER_HAND = 21
XYZ_PER_LANDMARK = 3
BASE_STATIC_FEATURES_PER_HAND = LANDMARKS_PER_HAND * XYZ_PER_LANDMARK  # 63
THUMB_FEATURES_PER_HAND = 6
STATIC_FEATURES_PER_HAND = BASE_STATIC_FEATURES_PER_HAND + THUMB_FEATURES_PER_HAND  # 69
DYNAMIC_FEATURES_PER_HAND = 3 + BASE_STATIC_FEATURES_PER_HAND
STATIC_FEATURE_COUNT = 2 * STATIC_FEATURES_PER_HAND  # 138
DYNAMIC_FEATURE_COUNT = 2 * DYNAMIC_FEATURES_PER_HAND  # 132
DYNAMIC_FRAMES = 24


def _angle(a, b, c):
    """Return angle ABC in radians, robust to degenerate landmarks."""
    ba = a - b
    bc = c - b
    denom = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom < 1e-8:
        return 0.0
    cosine = float(np.dot(ba, bc) / denom)
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _thumb_features(local):
    """Palm-scale-normalized distances and joint angles for thumb shape."""
    return [
        float(np.linalg.norm(local[4] - local[5])),  # tip to index MCP
        float(np.linalg.norm(local[4] - local[9])),  # tip to middle MCP
        float(np.linalg.norm(local[4] - local[8])),  # tip to index tip
        _angle(local[1], local[2], local[3]),
        _angle(local[2], local[3], local[4]),
        float(local[4, 1] - local[5, 1]),             # thumb tip y vs index MCP
    ]


def extract_features(results):
    """
    Return (static_features, dynamic_features, wrists_xy).

    Hand slots are ordered Left then Right when MediaPipe handedness is available.
    Static features are wrist-relative/palm-scale-normalized landmarks plus thumb
    geometry. Dynamic features retain absolute wrist xyz and normalized landmarks.
    Missing hands are zero-filled.
    """
    slots = {"Left": None, "Right": None}
    landmarks = getattr(results, "multi_hand_landmarks", None) or []
    handedness = getattr(results, "multi_handedness", None) or []

    for i, hand in enumerate(landmarks[:2]):
        side = None
        if i < len(handedness):
            try:
                side = handedness[i].classification[0].label
            except (AttributeError, IndexError):
                side = None
        if side not in ("Left", "Right"):
            side = "Left" if hand.landmark[0].x < 0.5 else "Right"
            if slots[side] is not None:
                side = "Right" if side == "Left" else "Left"
        if slots[side] is None:
            slots[side] = hand

    static_parts, dynamic_parts, wrists_xy = [], [], []
    for side in ("Left", "Right"):
        hand = slots[side]
        if hand is None:
            static_parts.extend([0.0] * STATIC_FEATURES_PER_HAND)
            dynamic_parts.extend([0.0] * DYNAMIC_FEATURES_PER_HAND)
            wrists_xy.append(None)
            continue

        raw = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark], dtype=np.float32)
        wrist = raw[0].copy()
        scale = max(float(np.linalg.norm(raw[9, :2] - wrist[:2])), 1e-4)
        local = np.clip((raw - wrist) / scale, -5.0, 5.0)

        static_parts.extend(local.flatten().tolist())
        static_parts.extend(_thumb_features(local))
        dynamic_parts.extend(wrist.tolist())
        dynamic_parts.extend(local.flatten().tolist())
        wrists_xy.append((float(wrist[0]), float(wrist[1])))

    return (
        np.asarray(static_parts, dtype=np.float32),
        np.asarray(dynamic_parts, dtype=np.float32),
        wrists_xy,
    )
