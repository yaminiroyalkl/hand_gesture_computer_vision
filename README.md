# hand_gesture_computer_vision

Real-time hand landmark detection using MediaPipe and OpenCV. Detects 21 hand landmarks per hand, draws skeleton overlay, and tracks hand movement across frames.

## What it does

- Captures live webcam feed
- Detects up to 2 hands simultaneously
- Identifies 21 landmark points (wrist, knuckles, fingertips) per hand
- Draws skeleton connections between landmarks
- Displays FPS for performance monitoring
- Packaged as a reusable module for other projects (gesture recognition, virtual controls, etc.)

## Tech

- Python 3.11
- MediaPipe (Tasks API) - hand landmark detection model
- OpenCV - camera access, image processing, drawing

## Project Structure

```
├── HandTrackingModule.py   # Reusable hand detection class
├── HandTrackingMin.py      # Minimal working example
├── Example.py              # Demo using the module
├── hand_landmarker.task    # Pre-trained model (Google MediaPipe)
├── NOTES.md                # Development learnings
└── README.md
```

## Usage

### As a module (recommended)

```python
import HandTrackingModule as htm

detector = htm.handDetector(maxHands=2, detectionCon=0.7)

# Inside your loop:
img = detector.findHands(img)              # detect + draw skeleton
lmList = detector.findPosition(img)        # get [id, x, y] for all landmarks

if len(lmList) != 0:
    thumb_tip = lmList[4]     # [4, px_x, px_y]
    index_tip = lmList[8]     # [8, px_x, px_y]
```

### Standalone

```bash
python HandTrackingMin.py
```

Press `q` to quit.

## Setup

```bash
pip install opencv-python mediapipe
```

Download the hand landmarker model:
```bash
curl -O https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

## Notes

- Uses the new MediaPipe Tasks API (0.10.30+). The old `mp.solutions` API is deprecated and removed.
- Requires Python 3.9-3.12 (mediapipe doesn't support 3.13+ yet).
- Only one application can access the webcam at a time.
