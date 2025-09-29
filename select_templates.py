import cv2
import os

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
TEMP_PATH = os.path.join(CURRENT_PATH, 'Templates_4')

# Zorg dat de map bestaat
os.makedirs(TEMP_PATH, exist_ok=True)

# Open video
cap = cv2.VideoCapture('card_video.mp4') 

if not cap.isOpened():
    print("Error opening video file")
    exit()

# Lees het eerste frame
ret, frame = cap.read()
if not ret:
    print("Kon geen frame lezen uit de video")
    cap.release()
    exit()

# Toon frame
cv2.imshow("Frame", frame)

# Laat gebruiker ROI selecteren
roi = cv2.selectROI("Select Template", frame, showCrosshair=True, fromCenter=False)
x, y, w, h = roi

# Sla ROI op als template
if w > 0 and h > 0:
    template = frame[y:y+h, x:x+w]
    cv2.imwrite(os.path.join(TEMP_PATH, "6.png"), template)
    print("Template saved!")

cv2.destroyAllWindows()
cap.release()
