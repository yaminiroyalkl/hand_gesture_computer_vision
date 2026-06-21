import cv2
import time
import HandTrackingModule as htm

pTime = 0
cap = cv2.VideoCapture(0)
detector = htm.handDetector()

while True:
    success, img = cap.read()
    if not success:
        break

    # Detect hands and draw skeleton (pass draw=False to hide)
    img = detector.findHands(img, draw=True)

    # Get landmark positions (pass draw=False to hide big circles)
    lmList = detector.findPosition(img, draw=False)

    # if len(lmList) != 0:
    #     print(lmList[4])# Prints [4, x, y] - Thumb tip pixel coordinates
    #     print(lmList[8])# Prints [8, x, y] - index tip pixel coordinates

    # FPS
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
