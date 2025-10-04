import cv2
import os

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
TEMP_PATH = os.path.join(CURRENT_PATH, 'Templates_UT_new')


os.makedirs(TEMP_PATH, exist_ok=True)
im = cv2.imread("pic_scard.jpg")

cv2.imshow("Frame",im)


# for i in [2,6,4,5,7,8]:
for i in [1]:
    scale = 0.5
    big_frame = cv2.resize(im, None, fx=scale, fy=scale)
    roi = cv2.selectROI("Select Template", big_frame, showCrosshair=True, fromCenter=False)
    x, y, w, h = roi
    x = int(x / scale)
    y = int(y / scale)
    w = int(w / scale)
    h = int(h / scale)




    if w > 0 and h > 0:
        template = im[y:y+h, x:x+w]
        # cv2.imwrite(os.path.join(TEMP_PATH, f"{i}.png"), template)
        cv2.imwrite(os.path.join(TEMP_PATH, "UT_temp.png"), template)

cv2.destroyAllWindows()