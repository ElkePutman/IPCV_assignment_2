import cv2
import numpy as np
from skimage import morphology
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageOps

class VideoProcessor:
    def __init__(self, input_file: str, output_file: str, down_fact: float = 1.0):
        self.input_file = input_file
        self.output_file = output_file
        self.cap = cv2.VideoCapture(input_file)
        self.fps = int(round(self.cap.get(cv2.CAP_PROP_FPS)))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.down_fact = down_fact
        self.new_width = int(self.width * down_fact)
        self.new_height = int(self.height * down_fact)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(output_file, fourcc, self.fps, (self.new_width, self.new_height))
        self.frame = None 
        self.current_time = None
        self.write_frame = True
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.end_time_vid = ((self.frame_count - 1) / self.fps) * 1000
        self.CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
        
        self.previous_frame = None
        self.p0 = None
        self.old_gray = None
        self.mask = None
        self.colors = None
        

        print('Processing new video')

    def __call__(self, show_video=False, debug_timestamp=None, save_debug_frame=False):
        """
        Maakt de VideoProcessor instantie callable.
        Als 'debug_timestamp' wordt opgegeven, wordt één frame verwerkt.
         """
        if debug_timestamp is not None:
            self.debug_single_frame(timestamp_ms=debug_timestamp, 
                                    show_video=show_video, 
                                    save_frame=save_debug_frame)
        else:
            self.run(show_video=show_video)


    def put_text(self,text, x=50, y=50, color=(255, 0, 0),sz_in = 0.7,inp=None,th_in = 2):

        sz = self.down_fact * sz_in
        th = int(self.down_fact * th_in)
        x_pos = int(self.down_fact * x)
        y_pos = int(self.down_fact * y)
        if inp is None:
            cv2.putText(self.frame, text, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, sz, color, th, cv2.LINE_AA)
        else:
            cv2.putText(inp, text, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, sz, color, th, cv2.LINE_AA)

    # helper function to change what you do based on video seconds
    def between(self, lower=None, upper=None) -> bool:
        if lower is not None and upper is not None:
            return lower <= int(self.cap.get(cv2.CAP_PROP_POS_MSEC)) < upper
        else:
            return self.lower <= int(self.cap.get(cv2.CAP_PROP_POS_MSEC)) < self.upper

    # downsample the video to a lower resolution
    def downsample(self):
        if (self.new_width, self.new_height) != (self.width, self.height):
            self.frame = cv2.resize(self.frame, (self.new_width, self.new_height))



    def temp_match(self,start_time,duration,**kwargs):
        end_time = start_time + duration - 1
        if not start_time <= self.current_time <= end_time:
            return
        temp_folder = kwargs.get("templates")
        TEMP_PATH = os.path.join(self.CURRENT_PATH,temp_folder)
        templates = os.listdir(TEMP_PATH)
        frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        all_loc = []
        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
            ]
        
        for i, template in enumerate(templates):
        # template = templates[4]
            temp_im = cv2.imread(os.path.join(TEMP_PATH,template),cv2.IMREAD_GRAYSCALE)
            h,w = temp_im.shape                
            res = cv2.matchTemplate(frame_gray,temp_im,cv2.TM_CCOEFF_NORMED)
            if kwargs.get("multiple"):            
                threshold = 0.87           
                loc = np.where( res >= threshold)
                all_loc.append(loc)
            else:
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                top_left = max_loc
                bottom_right = (top_left[0] + w, top_left[1] + h) 
                cv2.rectangle(self.frame,top_left, bottom_right, 255, 2)


        if kwargs.get("multiple"):
            for i in range(len(templates)):
                loc = all_loc[i]
                for pt in zip(*loc[::-1]):
                    cv2.rectangle(self.frame, pt, (pt[0] + w, pt[1] + h), colors[i], 2)

        self.put_text(kwargs.get("text"))

    # def KeyPoints_target(self,**kwargs):
    #     temp_folder = kwargs.get("templates")        
    #     TEMP_PATH = os.path.join(self.CURRENT_PATH,temp_folder)
    #     templates = os.listdir(TEMP_PATH)
    #     im = cv2.imread(os.path.join(self.TEMP_PATH,templates[0]),cv2.IMREAD_GRAYSCALE)
    #     sift = cv2.SIFT_create()
    #     keypoints_1, descriptors_1 = sift.detectAndCompute(im, None)
    #     roi_keypoint_image=cv2.drawKeypoints(im,keypoints_1,None)
    #     return

    # def SIFT(self,start_time,duration):
    #     end_time = start_time + duration - 1
    #     if not start_time <= self.current_time <= end_time:
    #         return
    #     gray_frame= cv2.cvtColor(self.frame,cv2.COLOR_BGR2GRAY)

    #     return

    def optical_flow(self,start_time,duration,**kwargs):
        end_time = start_time + duration - 1
        if not start_time <= self.current_time <= end_time:
            return
        # params for ShiTomasi corner detection
        feature_params = dict( maxCorners = 30,
                            qualityLevel = 0.5,
                            minDistance = 7)
        # Parameters for lucas kanade optical flow
        lk_params = dict( winSize  = (15, 15),
                        maxLevel = 2,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        # Take first frame and find corners in it        
        
        
        if self.p0 is None:
            self.old_gray = cv2.cvtColor(self.previous_frame, cv2.COLOR_BGR2GRAY)
            self.p0 = cv2.goodFeaturesToTrack(self.old_gray, mask = None, **feature_params)
            self.mask = np.zeros_like(self.previous_frame)
            self.colors = np.random.randint(0, 255, (100, 3))
        # Create a mask image for drawing purposes        
        frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.p0, None, **lk_params)

        if p1 is not None:
            good_new = p1[st==1]
            good_old = self.p0[st==1]

            # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            self.mask = cv2.line(self.mask, (int(a), int(b)), (int(c), int(d)), self.colors[i].tolist(), 2)
            self.frame = cv2.circle(self.frame, (int(a), int(b)), 5, self.colors[i].tolist(), -1)
        self.frame = cv2.add(self.frame, self.mask) 
        self.old_gray = frame_gray.copy()      
        self.p0 = good_new.reshape(-1, 1, 2)

        self.put_text(kwargs.get("text"))

 

    # def optical_flow_FB(self,start_time,duration):
    #     end_time = start_time + duration - 1
    #     if not start_time <= self.current_time <= end_time:
    #         return
    #     prev_gray = cv2.cvtColor(self.previous_frame, cv2.COLOR_BGR2GRAY)
    #     curr_gray = cv2.cvtColor(self.frame,cv2.COLOR_BGR2GRAY)

    #     #shi-Tomasi corner detection
    #     if self.corners is not None:
    #         self.corners= cv2.goodFeaturesToTrack(prev_gray,30,0.01,10) #gives coordinates x,y


    #     flow = cv2.calcOpticalFlowFarneback(prev_gray,curr_gray,None, 0.5, 3, 15, 3, 5, 1.2, 0) #is array in y,x

    #     for pt in corners:
    #         x,y = pt.ravel()
    #         dx, dy = flow[int(y),int(x)]
    #         print(dx)
    #         scale = 20
    #         cv2.arrowedLine(self.frame, (int(x), int(y)), (int(x + dx*scale), int(y + dy*scale)), (0, 0, 255), 2,tipLength = 0.5)
            


        

    #     return
        

    
        
            

   

    # run all the funtions
    def run(self, show_video=False):
        #list of exercises(fucntion, duration)
        exercises = [
            # (self.show_template, 6000),
            (self.temp_match, 20000,{"templates":"Numb_temp","multiple":True,"text":"Template matching of student numbers"}),
            (self.temp_match, 5000,{"templates":"UT_temp","text":"Template matching of student UT logo"}),
            (self.temp_match, 5000,{"templates":"photo_temp","text":"Template matching of photo"}),
            (self.temp_match, 5000,{"templates":"laptop_temp","text":"Template matching of laptop"}),
            (self.optical_flow,5000,{"text":"Optical flow"}),
            
        ]

        while self.cap.isOpened():
            ret, self.frame = self.cap.read()
            if not ret:
                break
            self.previous_frame = self.frame.copy()
            self.current_time = int(self.cap.get(cv2.CAP_PROP_POS_MSEC))
            self.downsample()

            start = 0


            for func, dur,kwargs in exercises:
                result = func(start, dur,**kwargs)
                start += dur
                
            
            self.out.write(self.frame)
            



            if show_video:
                cv2.imshow('Video', self.frame)
            # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break


    # if debugging is needed, run this
    def debug_single_frame(self, timestamp_ms, show_video=True, save_frame=False):

        self.cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
        ret, self.frame = self.cap.read()
        if not ret:
            print("Couldn't read the frame on timestamp", timestamp_ms)
            return
        
        self.current_time = int(self.cap.get(cv2.CAP_PROP_POS_MSEC))
        print(f"Debug frame on {self.current_time} ms")  

        
        

    
        exercises = [
            # (self.show_template, 6000),
            (self.temp_match, 20000,{"templates":"Templates_5","multiple":True}),
            (self.temp_match, 5000,{"templates":"Templates_UT"}),
            (self.temp_match, 5000,{"templates":"Templates_diff"}),
            (self.temp_match, 5000,{"templates":"Templates_laptop"}),
            (self.optical_flow_FB,5000,{}),
        ]

        start = 0

        for func, dur,kwargs in exercises:
            result = func(start, dur,**kwargs)
            start += dur
        
        self.out.write(self.frame)
        



        if show_video:
            cv2.imshow('Debug Frame', self.frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if save_frame:
            cv2.imwrite("debug_frame.png", self.frame)
            print("Saved frame")
            im = cv2.imread('debug_frame.png')
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            plt.figure()
            plt.imshow(im)
            plt.show()   
 
