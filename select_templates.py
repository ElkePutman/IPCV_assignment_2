import cv2
import os

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
TEMP_PATH = os.path.join(CURRENT_PATH, 'sift_diver')


os.makedirs(TEMP_PATH, exist_ok=True)


cap = cv2.VideoCapture('diver_slow.mp4') 

if not cap.isOpened():
    print("Error opening video file")
    exit()


cap.set(cv2.CAP_PROP_POS_MSEC, 0)
ret, frame = cap.read()
if not ret:
    print("Couldn't read the frame")
    cap.release()
    exit()


cv2.imshow("Frame", frame)


# for i in [2,6,4,5,7,8]:
for i in [1]:
    scale = 1
    big_frame = cv2.resize(frame, None, fx=scale, fy=scale)
    roi = cv2.selectROI("Select Template", big_frame, showCrosshair=True, fromCenter=False)
    x, y, w, h = roi
    x = int(x / scale)
    y = int(y / scale)
    w = int(w / scale)
    h = int(h / scale)

    if w > 0 and h > 0:
        template = frame[y:y+h, x:x+w]
        # cv2.imwrite(os.path.join(TEMP_PATH, f"{i}.png"), template)
        cv2.imwrite(os.path.join(TEMP_PATH, "dive_bottle_2.png"), template)

cv2.destroyAllWindows()
cap.release()


