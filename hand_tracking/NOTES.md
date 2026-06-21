# Learnings - Hand Tracking Project

## What I built
Real-time hand landmark detection using webcam. Detects 21 points on each hand, draws skeleton overlay, tracks movement across frames. Packaged as a reusable module.

---

## Key Concepts I Picked Up

### MediaPipe's Detection Pipeline
MediaPipe uses a 2-stage approach internally:
1. Palm detection - finds rough bounding box of where hand is in the frame
2. Landmark detection - within that box, locates 21 specific points

This is faster than scanning the entire image for fine-grained points directly.

### The New Tasks API (post 0.10.30)
The old `mp.solutions` API was removed entirely. The new pattern is:

```
model file ──→ BaseOptions ──┐
                             ├──→ HandLandmarkerOptions ──→ HandLandmarker
running_mode ────────────────┘         │                        │
                                       │                        │
camera frame ──→ mp.Image ─────────────┘                        │
                                                                ▼
                                                    HandLandmarkerResult
                                                    ├── hand_landmarks (21 points × N hands)
                                                    ├── hand_world_landmarks (3D coords)
                                                    └── handedness (Left/Right)
```

Every vision task in mediapipe (hands, face, pose, objects) follows this exact same structure. Only the model file and task-specific options change.

### Running Modes
- **IMAGE** - single photo, `detect(image)`, no timestamp needed
- **VIDEO** - continuous frames, `detect_for_video(image, timestamp_ms)`, uses timestamps for temporal smoothing
- **LIVE_STREAM** - async non-blocking, `detect_async(image, timestamp_ms)`, fires callback

Used VIDEO mode here since we're processing webcam frames sequentially.

### Confidence Thresholds
Three knobs to tune detection behavior:
- **detection_confidence** - how sure the model needs to be that something is a hand (lower = more sensitive, more false positives)
- **presence_confidence** - between frames, how sure it needs to be that the hand is still there before triggering re-detection
- **tracking_confidence** - how confidently it can track the same hand across frames vs re-detecting from scratch

### Normalized Coordinates
Landmarks come as 0.0-1.0 values, not pixel positions. This makes it resolution-independent. To get actual pixel coords:
```python
cx = int(lm.x * img_width)
cy = int(lm.y * img_height)
```

### BGR vs RGB
OpenCV reads in BGR, MediaPipe expects RGB. Must convert with `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` before feeding frames to the detector.

### Timestamps
VIDEO mode needs monotonically increasing timestamps (milliseconds) for temporal smoothing and tracking optimization. Used `int(time.time() * 1000)`.

---

## The 21 Landmarks

```
WRIST = 0
THUMB:  CMC=1, MCP=2, IP=3, TIP=4
INDEX:  MCP=5, PIP=6, DIP=7, TIP=8
MIDDLE: MCP=9, PIP=10, DIP=11, TIP=12
RING:   MCP=13, PIP=14, DIP=15, TIP=16
PINKY:  MCP=17, PIP=18, DIP=19, TIP=20
```

```
        8   12  16  20      ← fingertips
        |   |   |   |
    4   7   11  15  19
    |   |   |   |   |
    3   6   10  14  18
    |   |   |   |   |
    2   5---9---13--17      ← knuckles + palm
    |   |
    1   |
     \ /
      0                     ← wrist
```

### Gesture Detection Logic
- Finger UP: TIP.y < PIP.y (y=0 is top of image)
- Finger DOWN: TIP.y > PIP.y
- Thumb: compare TIP.x vs MCP.x (horizontal movement, not vertical)

---

## Hand Connections (Skeleton)

```python
HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),         # thumb
    (0,5), (5,6), (6,7), (7,8),         # index
    (0,9), (9,10), (10,11), (11,12),    # middle
    (0,13), (13,14), (14,15), (15,16),  # ring
    (0,17), (17,18), (18,19), (19,20),  # pinky
    (5,9), (9,13), (13,17)              # palm
]
```

In the old API, `mpDraw.draw_landmarks()` handled this automatically. Now we loop through connections and draw with `cv2.line()` manually.

---

## FPS Calculation

```python
pTime = 0
# inside loop:
cTime = time.time()
fps = 1 / (cTime - pTime)
pTime = cTime
```

Simple: 1 divided by time-between-frames = frames per second. Useful for monitoring if detection is too heavy.

---

## Module Pattern

Extracted the detection logic into `HandTrackingModule.py` as a reusable class:

```python
class handDetector:
    __init__()      # setup model, options, create detector
    findHands()     # detect + draw skeleton, returns img
    findPosition()  # returns list of [id, x, y] for all landmarks
```

Usage from any file:
```python
import HandTrackingModule as htm
detector = htm.handDetector()
img = detector.findHands(img)
lmList = detector.findPosition(img)
print(lmList[4])  # thumb tip coords
```

`if __name__ == "__main__":` at the bottom ensures the test code only runs when executing the module directly, not when importing it.

---

## Old API vs New API

| Concept | Old (removed in 0.10.30+) | New (current) |
|---------|--------------------------|---------------|
| Import | `mp.solutions.hands` | `mp.tasks.vision.HandLandmarker` |
| Drawing | `mp.solutions.drawing_utils` | Manual cv2.line/circle |
| Create detector | `Hands()` | `HandLandmarker.create_from_options()` |
| Process | `hands.process(imgRGB)` | `detector.detect_for_video(mp_image, ts)` |
| Results | `results.multi_hand_landmarks` | `result.hand_landmarks` |
| Draw skeleton | `mpDraw.draw_landmarks(...)` | Loop through connections + cv2 |

Most tutorials online (pre-2025) use the old API. Had to migrate everything to the tasks-based API.

---

## Useful Patterns

Detect which fingers are up:
```python
fingers = [
    points[4][0] < points[3][0],    # thumb (x-axis)
    points[8][1] < points[6][1],    # index
    points[12][1] < points[10][1],  # middle
    points[16][1] < points[14][1],  # ring
    points[20][1] < points[18][1],  # pinky
]
```

Distance between two landmarks (pinch detection):
```python
import math
dist = math.sqrt((points[8][0]-points[4][0])**2 + (points[8][1]-points[4][1])**2)
```

---

## References
- MediaPipe Tasks docs: https://ai.google.dev/edge/mediapipe/solutions
- Model download: https://storage.googleapis.com/mediapipe-models/
