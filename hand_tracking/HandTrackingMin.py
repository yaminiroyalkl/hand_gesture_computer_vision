import cv2
import mediapipe as mp
import time

# New mediapipe API setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Hand connections - which landmarks connect to which (same as old HAND_CONNECTIONS)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17)              # Palm
]

# Create hand landmarker
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=None                
)

cap = cv2.VideoCapture(0)
hand_landmarker = HandLandmarker.create_from_options(options)
pTime = 0  # previous time for FPS calculation

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read from camera")
        break

    # Convert to mediapipe image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    # Detect hands
    timestamp_ms = int(time.time() * 1000)
    result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

    #print(result.hand_landmarks)

    # Draw hand landmarks and connections
    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            h, w, c = img.shape

            # Convert all landmarks to pixel coordinates
            points = []
            for id, lm in enumerate(hand_landmarks):
                cx, cy = int(lm.x * w), int(lm.y * h)
                points.append((cx, cy))

                # Print only fingertips for debugging
                if id in [4, 8, 12, 16, 20]:
                    print(f"Landmark {id}: pixel ({cx}, {cy})")

            # Highlight a specific landmark (e.g., id=4 thumb tip) with bigger circle
            cv2.circle(img, points[4], 15, (255, 0, 0), cv2.FILLED)  # Blue - Thumb tip
            cv2.circle(img, points[8], 15, (255, 0, 0), cv2.FILLED)  # Blue - index tip

            # Draw lines between connected landmarks
            for connection in HAND_CONNECTIONS:
                start = points[connection[0]]
                end = points[connection[1]]
                cv2.line(img, start, end, (0, 255, 0), 2)  # Green lines

            # Draw dots on top of lines
            for cx, cy in points:
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)  # Pink dots

    # FPS calculation
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
