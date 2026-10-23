"""Collect labelled static-pose and dynamic-motion samples from a webcam.

Examples:
  python collect_data.py --mode static --label OPEN_PALM
  python collect_data.py --mode dynamic --label WAVE

Static: press S to save a sample; change hand pose between saves.
Dynamic: press SPACE to record one 24-frame movement sample.
Press Q or ESC to quit.
"""
import argparse
import csv
import os
import time

import cv2
import mediapipe as mp
import numpy as np

from features import (
    extract_features, STATIC_FEATURE_COUNT, DYNAMIC_FEATURE_COUNT,
    DYNAMIC_FRAMES
)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
STATIC_FILE = os.path.join(DATA_DIR, "static_samples.csv")
DYNAMIC_FILE = os.path.join(DATA_DIR, "dynamic_samples.csv")


def append_csv(path, label, features):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["label"] + [f"f{i}" for i in range(len(features))])
        writer.writerow([label] + [f"{float(v):.7f}" for v in features])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("static", "dynamic"), required=True)
    parser.add_argument("--label", required=True, help="Exact class label, e.g. OPEN_PALM or WAVE")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise SystemExit("Could not open webcam. Try --camera 1.")

    dynamic_buffer = []
    recording = False
    saved = 0
    dataset_path = STATIC_FILE if args.mode == "static" else DYNAMIC_FILE
    print(f"Dataset path: {dataset_path}")
    with mp_hands.Hands(
        static_image_mode=False, max_num_hands=2, model_complexity=0,
        min_detection_confidence=0.6, min_tracking_confidence=0.6
    ) as hands:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Camera frame could not be read.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            for hand in result.multi_hand_landmarks or []:
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            static_features, dynamic_features, _ = extract_features(result)
            if args.mode == "dynamic" and recording:
                dynamic_buffer.append(dynamic_features)
                if len(dynamic_buffer) >= DYNAMIC_FRAMES:
                    flattened = np.asarray(dynamic_buffer[:DYNAMIC_FRAMES], dtype=np.float32).flatten()
                    append_csv(DYNAMIC_FILE, args.label, flattened)
                    saved += 1
                    print(f"Saved dynamic sample {saved}: {args.label}")
                    dynamic_buffer.clear()
                    recording = False

            lines = [
                f"MODE: {args.mode.upper()}   LABEL: {args.label}",
                f"Saved this session: {saved}",
                "Static: press S to save one frame; vary pose and position",
                f"Dynamic: SPACE records {DYNAMIC_FRAMES} frames",
                "Q / ESC: quit"
            ]
            if args.mode == "dynamic":
                lines.append(f"Recording: {len(dynamic_buffer)}/{DYNAMIC_FRAMES}" if recording else "Ready for motion sample")
            for i, line in enumerate(lines):
                cv2.putText(frame, line, (12, 25 + i * 27), cv2.FONT_HERSHEY_SIMPLEX,
                            0.62, (40, 240, 240), 2, cv2.LINE_AA)
            cv2.imshow("Gesture Data Collector", frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                break
            if args.mode == "static" and key == ord("s"):
                if not (result.multi_hand_landmarks or []):
                    print("No hand detected; sample not saved.")
                else:
                    append_csv(STATIC_FILE, args.label, static_features)
                    saved += 1
                    print(f"Saved static sample {saved}: {args.label}")
                    time.sleep(0.08)
            if args.mode == "dynamic" and key == 32 and not recording:
                if not (result.multi_hand_landmarks or []):
                    print("No hand detected at start; sample not recorded.")
                else:
                    dynamic_buffer.clear()
                    dynamic_buffer.append(dynamic_features)
                    recording = True
                    print(f"Recording {args.label}... perform the movement now.")

    camera.release()
    cv2.destroyAllWindows()
    print(f"Finished. Session samples saved: {saved}")


if __name__ == "__main__":
    main()
