# %%
# test orb keypoints
# Import necessary libraries
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# %%
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
TEMP_PATH = os.path.join(CURRENT_PATH,"sift_diver")
templates = os.listdir(TEMP_PATH)

# im_temp = cv2.imread(os.path.join(TEMP_PATH,templates[0]))
im_temp = cv2.imread(os.path.join(TEMP_PATH,'dive_bottle.png'))
# print(im_temp.shape)
#%%
# angle = -18  # graden (negatief = rechtsom)
# (h, w) =im_temp.shape[:2]
# center = (w // 2, h // 2)

# M = cv2.getRotationMatrix2D(center, angle, 1.0)
# im_temp = cv2.warpAffine(im_temp, M, (w, h))

# plt.imshow(im_temp)
# plt.show()
#%%

# cv2.imwrite(os.path.join(TEMP_PATH, "dive_bottle_rotated.png"), rotated)



im_temp_gray = cv2.cvtColor(im_temp,cv2.COLOR_BGR2GRAY)
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(im_temp_gray, None)
roi_keypoint_image=cv2.drawKeypoints(im_temp,kp1, None)
# visualize key points 
# plt.subplot(121)
# plt.imshow(im_temp,cmap="gray")

# plt.subplot(122)
# plt.imshow(roi_keypoint_image,cmap="gray")
# plt.show()

# %%

cap = cv2.VideoCapture('diver_slow.mp4')

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
input_fps = cap.get(cv2.CAP_PROP_FPS)
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_diver.mp4', fourcc, input_fps, (frame_width, frame_height))

if not cap.isOpened():
    print("Error opening video file")
    exit()


# cap.set(cv2.CAP_PROP_POS_MSEC, 1000)
while True:
    ret, frame = cap.read()

    if not ret:
        print("Couldn't read the frame")
        cap.release()
        exit()

    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)


    kp2, des2 = orb.detectAndCompute(frame_gray, None)



    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True) #create object 
    matches = bf.match(des1,des2)

    sorted_matches = sorted(matches, key=lambda x: x.distance)
    # good = [m for m in matches if m.distance < 0.75 * matches[0].distance]
    good = sorted_matches[:50]

    MIN_MATCH_COUNT = 20
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

    # else:
    #     print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    #     matchesMask = None
        if H is not None:
            h, w = im_temp_gray.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, H)


            draw_params = dict(matchColor = (0,255,0), 
                            singlePointColor = None,
                            matchesMask = matchesMask, 
                            flags = 2)
            img3 = cv2.drawMatches(im_temp,kp1,frame,kp2,good,None,**draw_params)
            # plt.imshow(img3, 'gray'),plt.show()

            frame_vis = cv2.polylines(frame.copy(), [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

            new_center = np.mean(dst[:, 0, :], axis=0)  # [x, y]

            alpha = 0.9  
            if 'smooth_center' not in locals():
                smooth_center = new_center
            else:
                smooth_center = alpha * smooth_center + (1 - alpha) * new_center

            x, y = smooth_center  
            # x, y = np.mean(dst[:,0,0]), np.mean(dst[:,0,1])
            info_text = f"Air: {150 :.0f} bar"
            depth = random.uniform(7,8)
            depth_text = f"Depth: {depth:.1f} m"
            cv2.putText(frame, info_text, (int(x)-100, int(y)-50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(frame, depth_text, (int(x)-100, int(y)-20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)  
    

    out.write(frame)

# Release the capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows() 



