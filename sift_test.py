# test orb keypoints
# Import necessary libraries
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import os

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
TEMP_PATH = os.path.join(CURRENT_PATH,"Templates_UT")
templates = os.listdir(TEMP_PATH)

im_temp = cv2.imread(os.path.join(TEMP_PATH,templates[0]),cv2.IMREAD_GRAYSCALE)
orb = cv2.ORB_create()
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(im_temp, None)
roi_keypoint_image=cv2.drawKeypoints(im_temp,kp1, None)
# visualize key points 
plt.subplot(121)
plt.imshow(im_temp,cmap="gray")

plt.subplot(122)
plt.imshow(roi_keypoint_image,cmap="gray")
plt.show()

bf = cv2.BFMatcher()
im_frame = cv2.imread('debug_frame.png',cv2.IMREAD_GRAYSCALE)

kp2, des2 = sift.detectAndCompute(im_frame, None)
# des1 = des1.astype('float32')
# des2 = des2.astype('float32')


# matches = bf.knnMatch(des1, des2, k=2)

# good = []
# for m,n in matches:
#     if m.distance < 0.75*n.distance:
#         good.append(m)

matches = bf.knnMatch(des1,des2,k = 2)

# Lowe's ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append(m) 

 
# Draw first 10 matches.
img3 = cv2.drawMatches(im_temp,kp1,im_frame,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
 
plt.imshow(img3),plt.show()

MIN_MATCH_COUNT =10

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,1.0)
    matchesMask = mask.ravel().tolist()
    h,w = im_temp.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    im_frame = cv2.polylines(im_frame,[np.int32(dst)],True,255,3, cv2.LINE_AA)
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                singlePointColor = None,
                matchesMask = matchesMask, # draw only inliers
                flags = 2)
img3 = cv2.drawMatches(im_temp,kp1,im_frame,kp2,good,None,**draw_params)
plt.imshow(img3, 'gray'),plt.show()

import numpy as np
import cv2

# Neem alleen keypoints die bij inliers horen
inlier_kp1 = [kp1[m.queryIdx] for i, m in enumerate(good) if matchesMask[i]]
inlier_kp2 = [kp2[m.trainIdx] for i, m in enumerate(good) if matchesMask[i]]



# Voor frame (im_frame)
pts2 = np.array([kp.pt for kp in inlier_kp2], dtype=np.int32)
x2, y2, w2, h2 = cv2.boundingRect(pts2)
cv2.rectangle(im_frame, (x2, y2), (x2+w2, y2+h2), (0,0,255), 2)  # rode rechthoek



plt.imshow(im_frame)
plt.show()



          
