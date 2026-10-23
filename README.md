# Trainable Gesture Interpreter

This project supports:
- Trained static hand poses, including two-hand combinations.
- Trained movement gestures using a sequence of 24 frames.
- Automatic speech after a gesture is accepted.
- Exporting trained models to `models/` and copying them to another computer.

## 1. Install

Recommended: 64-bit Python 3.11 on Windows.

```bash
python -m venv venv
```

Git Bash:
```bash
source venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. Collect static-pose samples

Run one collector per label. Press `S` to save a sample, change pose/angle/distance, and repeat. Press `Q` to exit.

```bash
python collect_data.py --mode static --label OPEN_PALM
python collect_data.py --mode static --label CLOSED_FIST
python collect_data.py --mode static --label TWO_FINGERS
python collect_data.py --mode static --label POINTING
python collect_data.py --mode static --label THUMB_UP
python collect_data.py --mode static --label THUMB_DOWN
python collect_data.py --mode static --label THREE_FINGERS
python collect_data.py --mode static --label PINKY_UP
python collect_data.py --mode static --label ROCK_ON
python collect_data.py --mode static --label OK_SIGN
python collect_data.py --mode static --label TWO_OPEN_PALMS
python collect_data.py --mode static --label TWO_FISTS
python collect_data.py --mode static --label OPEN_PALM_AND_FIST
python collect_data.py --mode static --label TWO_THUMBS_UP
```

Start with 100–200 samples per class. Capture different orientations, distances, lighting and backgrounds. For two-hand labels, keep both hands visible and vary their spacing and relative positions.

## 3. Collect movement samples

For each dynamic label, run the collector and press SPACE, then perform the motion while it records 24 frames. Each SPACE press saves one sample.

```bash
python collect_data.py --mode dynamic --label WAVE
python collect_data.py --mode dynamic --label MOVE_HAND_LEFT
python collect_data.py --mode dynamic --label MOVE_HAND_RIGHT
python collect_data.py --mode dynamic --label BOTH_HANDS_UP
```

Collect at least 100–200 clips per movement class, with varied speed, distance and starting position. Perform the motion only after pressing SPACE. Include similar-looking negative examples by collecting other motion classes.

You can change labels and spoken words in `labels.json`. A label used in the collector must match a label used in `labels.json` for the desired spoken word.

Samples append to `data/static_samples.csv` and `data/dynamic_samples.csv`. Back up these files. If you change feature extraction in `features.py`, recollect the dataset rather than mixing incompatible feature formats.

## 4. Train and export

Once you have at least two classes with at least two samples per class (more data is strongly recommended):

```bash
python train_model.py
```

Output files:
- `models/static_model.joblib`
- `models/dynamic_model.joblib`

Training prints a held-out classification report. The random split is a preliminary check; for a stronger estimate, collect a separate test session and evaluate on it.

## 5. Run the interpreter

```bash
python main.py
```

It loads whichever trained model files exist. If a model is missing, that mode is unavailable until trained. The speech engine speaks accepted predictions automatically.

## 6. Transfer to another computer

Copy these to the target computer:
- `main.py`
- `features.py`
- `labels.json`
- `requirements.txt`
- `models/static_model.joblib`
- `models/dynamic_model.joblib`

Install dependencies using `pip install -r requirements.txt`, then run `python main.py`. The training CSVs and training script are not required for inference unless you want to retrain.

Only load model files you created or trust. Serialized joblib files can execute code when loaded.

## Notes / limitations

- This is a starting pipeline, not a production sign-language translator.
- The static model learns pose geometry, not formal sign language.
- The dynamic model uses a fixed 24-frame window and a simple movement gate; tune `DYNAMIC_MOTION_THRESHOLD`, confidence thresholds and cooldown for your camera and gestures.
- MediaPipe handedness can be less reliable if the camera image is not mirrored consistently. Keep the same mirrored-camera convention during collection and runtime.
- Two-hand samples must include both hands; one-hand samples are zero-padded for the absent hand.
- Avoid class imbalance. Collect similar numbers of samples for every label.
