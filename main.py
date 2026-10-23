"""Runtime app: trained static and dynamic gesture models + automatic speech."""
import json
import os
import queue
import threading
import time
from collections import deque
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import joblib
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

from features import (
    extract_features, STATIC_FEATURE_COUNT, DYNAMIC_FEATURE_COUNT, DYNAMIC_FRAMES
)

MODEL_DIR = "models"
LABELS_FILE = "labels.json"
CONFIRM_SECONDS = 0.55
DYNAMIC_COOLDOWN_SECONDS = 1.4
DYNAMIC_STRIDE = 8
DYNAMIC_MOTION_THRESHOLD = 0.012


def load_labels():
    with open(LABELS_FILE, encoding="utf-8") as f:
        data = json.load(f)
    return {
        item["label"]: item["word"]
        for group in ("static", "dynamic")
        for item in data.get(group, [])
    }


class InterpreterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Trained Gesture Interpreter")
        self.root.geometry("1050x720")
        self.root.minsize(800, 580)
        self.root.configure(bg="#101827")

        self.running = True
        self.camera = None
        self.hands = None
        self.photo = None

        self.labels = load_labels()
        self.static_bundle = self.load_model("static_model.joblib", STATIC_FEATURE_COUNT)
        self.dynamic_bundle = self.load_model(
            "dynamic_model.joblib", DYNAMIC_FRAMES * DYNAMIC_FEATURE_COUNT
        )

        self.candidate = None
        self.candidate_since = 0.0
        self.last_added = None
        self.current_word = ""
        self.sentence = []
        self.dynamic_frames = deque(maxlen=DYNAMIC_FRAMES)
        self.dynamic_frame_count = 0
        self.last_dynamic_time = 0.0
        self.last_dynamic_label = None

        self.speech_queue = queue.Queue()
        self.speech_thread = None
        if pyttsx3 is not None:
            self.speech_thread = threading.Thread(target=self.speech_worker, daemon=True)
            self.speech_thread.start()

        self.build_ui()
        self.start_camera()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(30, self.update_frame)

    def load_model(self, filename, expected_features):
        path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(path):
            return None
        try:
            bundle = joblib.load(path)
            if bundle.get("feature_count") != expected_features:
                print(f"Ignoring incompatible model: {path}")
                return None
            return bundle
        except Exception as error:
            print(f"Could not load {path}: {error}")
            return None

    def build_ui(self):
        tk.Label(self.root, text="TRAINED GESTURE INTERPRETER",
                 font=("Segoe UI", 22, "bold"), fg="#67E8F9",
                 bg="#101827").pack(pady=(14, 3))
        tk.Label(self.root, text="STATIC + DYNAMIC + TWO-HAND RECOGNITION",
                 font=("Segoe UI", 10), fg="#B8C5D8",
                 bg="#101827").pack(pady=(0, 12))

        main = tk.Frame(self.root, bg="#101827")
        main.pack(fill="both", expand=True, padx=16, pady=4)
        left = tk.Frame(main, bg="#1E293B")
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))
        tk.Label(left, text="LIVE CAMERA", font=("Segoe UI", 13, "bold"),
                 fg="white", bg="#1E293B").pack(pady=9)
        self.video = tk.Label(left, text="Starting camera...", bg="#0B1120", fg="white")
        self.video.pack(fill="both", expand=True, padx=10, pady=(0, 8))
        self.camera_status = tk.Label(left, text="Checking camera...",
                                      fg="#FBBF24", bg="#1E293B")
        self.camera_status.pack(pady=(0, 9))

        right = tk.Frame(main, bg="#1E293B", width=290)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)
        tk.Label(right, text="RECOGNIZED WORD", font=("Segoe UI", 11, "bold"),
                 fg="#B8C5D8", bg="#1E293B").pack(pady=(20, 5))
        self.word_label = tk.Label(right, text="—", font=("Segoe UI", 23, "bold"),
                                   fg="#34D399", bg="#1E293B", wraplength=260)
        self.word_label.pack(pady=6)
        self.status_label = tk.Label(right, text="Show a trained gesture",
                                     font=("Segoe UI", 10), fg="#B8C5D8",
                                     bg="#1E293B", wraplength=260)
        self.status_label.pack(pady=8)
        model_status_text = (
            "Static model: " + ("ready" if self.static_bundle else "missing") +
            "\nDynamic model: " + ("ready" if self.dynamic_bundle else "missing")
        )
        self.model_status = tk.Label(
            right, text=model_status_text,
            font=("Segoe UI", 9), fg="#FBBF24", bg="#1E293B", wraplength=260
        )
        self.model_status.pack(pady=8)

        bottom = tk.Frame(self.root, bg="#1E293B")
        bottom.pack(fill="x", padx=16, pady=(8, 10))
        tk.Label(bottom, text="BUILT SENTENCE", font=("Segoe UI", 11, "bold"),
                 fg="#67E8F9", bg="#1E293B").pack(anchor="w", padx=12, pady=(8, 2))
        self.sentence_label = tk.Label(
            bottom, text="Your translated words will appear here.",
            font=("Segoe UI", 13), fg="white", bg="#1E293B",
            wraplength=950, justify="left"
        )
        self.sentence_label.pack(fill="x", padx=12, pady=5)
        controls = tk.Frame(bottom, bg="#1E293B")
        controls.pack(pady=(3, 9))
        ttk.Button(controls, text="Clear Sentence", command=self.clear_sentence).pack(side="left", padx=5)
        ttk.Button(controls, text="Exit", command=self.close).pack(side="left", padx=5)

    def start_camera(self):
        self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.camera.isOpened():
            self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            self.camera_status.config(text="Camera unavailable", fg="#F87171")
            return
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=2, model_complexity=0,
            min_detection_confidence=0.6, min_tracking_confidence=0.6
        )
        self.camera_status.config(text="Camera connected", fg="#34D399")

    def speech_worker(self):
        try:
            engine = pyttsx3.init()
        except Exception as error:
            print(f"Speech engine initialization failed: {error}")
            return
        while True:
            word = self.speech_queue.get()
            try:
                if word is None:
                    return
                engine.say(word)
                engine.runAndWait()
            except Exception as error:
                print(f"Speech error: {error}")
            finally:
                self.speech_queue.task_done()

    def speak(self, word):
        if pyttsx3 is not None and word:
            self.speech_queue.put(word)

    def update_frame(self):
        if not self.running:
            return
        if self.camera is None or not self.camera.isOpened() or self.hands is None:
            self.root.after(250, self.update_frame)
            return
        ok, frame = self.camera.read()
        if not ok:
            self.root.after(50, self.update_frame)
            return

        frame = cv2.flip(frame, 1)
        result = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        for hand in result.multi_hand_landmarks or []:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand, mp.solutions.hands.HAND_CONNECTIONS
            )

        static_features, dynamic_features, wrists = extract_features(result)
        self.process_static(static_features)
        self.process_dynamic(dynamic_features, wrists)

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image.thumbnail((640, 420))
        self.photo = ImageTk.PhotoImage(image=image)
        self.video.configure(image=self.photo, text="")
        self.root.after(15, self.update_frame)

    def process_static(self, features):
        if self.static_bundle is None:
            return
        if not any(features):
            self.candidate = None
            self.last_added = None
            return
        model = self.static_bundle["model"]
        probabilities = model.predict_proba(features.reshape(1, -1))[0]
        index = int(np.argmax(probabilities))
        label = str(model.classes_[index])
        confidence = float(probabilities[index])
        if confidence < 0.68:
            self.status_label.config(text=f"Uncertain static pose ({confidence:.0%})")
            self.candidate = None
            self.last_added = None
            return
        now = time.monotonic()
        if label != self.candidate:
            self.candidate = label
            self.candidate_since = now
        if now - self.candidate_since < CONFIRM_SECONDS:
            self.status_label.config(text=f"Confirming {label.replace('_', ' ')} ({confidence:.0%})")
            return
        word = self.labels.get(label, label.replace("_", " "))
        self.word_label.config(text=word)
        self.status_label.config(text=f"Static: {label} ({confidence:.0%})")
        if self.last_added != label:
            self.sentence.append(word)
            self.last_added = label
            self.update_sentence()
            self.speak(word)

    def process_dynamic(self, features, wrists):
        if self.dynamic_bundle is None:
            return
        self.dynamic_frames.append(features)
        self.dynamic_frame_count += 1
        if len(self.dynamic_frames) < DYNAMIC_FRAMES:
            return
        if self.dynamic_frame_count % DYNAMIC_STRIDE:
            return

        # Require visible wrist movement over the current window to reduce
        # repeated dynamic predictions while a hand is held still.
        wrist_series = np.array([
            [v[0], v[1], v[2], v[66], v[67], v[68]]
            for v in self.dynamic_frames
        ], dtype=np.float32)
        motion = float(np.mean(np.std(wrist_series[:, [0, 1, 3, 4]], axis=0)))
        if motion < DYNAMIC_MOTION_THRESHOLD:
            return

        vector = np.asarray(self.dynamic_frames, dtype=np.float32).flatten().reshape(1, -1)
        model = self.dynamic_bundle["model"]
        probabilities = model.predict_proba(vector)[0]
        index = int(np.argmax(probabilities))
        label = str(model.classes_[index])
        confidence = float(probabilities[index])
        now = time.monotonic()
        if confidence < 0.70 or now - self.last_dynamic_time < DYNAMIC_COOLDOWN_SECONDS:
            return
        if label == self.last_dynamic_label and now - self.last_dynamic_time < 2.5:
            return

        word = self.labels.get(label, label.replace("_", " "))
        self.last_dynamic_label = label
        self.last_dynamic_time = now
        self.word_label.config(text=word)
        self.status_label.config(text=f"Dynamic: {label} ({confidence:.0%})")
        self.sentence.append(word)
        self.update_sentence()
        self.speak(word)

    def update_sentence(self):
        self.sentence_label.config(
            text=" ".join(self.sentence) or "Your translated words will appear here."
        )

    def clear_sentence(self):
        self.sentence.clear()
        self.update_sentence()

    def close(self):
        self.running = False
        if self.camera is not None:
            self.camera.release()
        if self.hands is not None:
            self.hands.close()
        if self.speech_thread is not None:
            self.speech_queue.put(None)
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = InterpreterApp(root)
    root.mainloop()
