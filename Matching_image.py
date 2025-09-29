import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 as cv
import os
from PIL import Image, ImageOps

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
TEMP_PATH = os.path.join(CURRENT_PATH,'Templates_4')
templates = os.listdir(TEMP_PATH)

im = Image.open('debug_frame.png')
# plt.figure()
# plt.imshow(im)
# plt.axis('off')
# plt.show()

img = cv.imread('debug_frame.png')

# for template in templates:
template = templates[1]
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
temp_im = cv.imread(os.path.join(TEMP_PATH,template),cv.IMREAD_GRAYSCALE)
h,w = temp_im.shape 
res = cv.matchTemplate(img_gray,temp_im,cv.TM_CCOEFF_NORMED)
# plt.figure()
# plt.imshow(res)
# plt.axis('off')
# plt.show()
threshold = 0.8
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

print(img.shape)
    
cv.imwrite('res.png',img)








