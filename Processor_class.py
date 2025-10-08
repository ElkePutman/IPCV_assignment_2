import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random

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


        self.corners = None

        self.orb = None
        self.kp1 = None
        self.des1 = None
        self.im_temp_gray = None
        self.bf = None
        self.smooth_center = None
        self.frame_number = 0
        self.depth = None
        

        print('Processing new video')

    def __call__(self, show_video=False, debug_timestamp=None, save_debug_frame=False):
        if debug_timestamp is not None:
            self.debug_single_frame(timestamp_ms=debug_timestamp, 
                                    show_video=show_video, 
                                    save_frame=save_debug_frame)
        else:
            self.run(show_video=show_video)


    # Put text to video frames
    def put_text(self,text, x=50, y=50, color=(255, 0, 0),sz_in = 0.7,inp=None,th_in = 2):
        sz = self.down_fact * sz_in
        th = int(self.down_fact * th_in)
        x_pos = int(self.down_fact * x)
        y_pos = int(self.down_fact * y)
        if inp is None:
            cv2.putText(self.frame, text, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, sz, color, th, cv2.LINE_AA)
        else:
            cv2.putText(inp, text, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, sz, color, th, cv2.LINE_AA)


    # downsample the video to a lower resolution
    def downsample(self):
        if (self.new_width, self.new_height) != (self.width, self.height):
            self.frame = cv2.resize(self.frame, (self.new_width, self.new_height))


    # Place templates on the frames
    def overlay_template_on_frame(self, template_path,margin2,scale = 4):        
        temp_color = cv2.imread(template_path, cv2.IMREAD_COLOR)
        if temp_color is None:
            raise FileNotFoundError(f"Template niet gevonden: {template_path}")

        temp_color = cv2.resize(temp_color, None, fx=scale, fy=scale)
        th, tw = temp_color.shape[:2]
        fh, fw = self.frame.shape[:2]

        margin1 = fw-tw-20
          

        x,y = int(margin1),int(margin2)  

        self.frame[y:y+th, x:x+tw] = temp_color
        return self.frame

    # Template matching of several templates
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
        
        margin2 = 10                    
        scale = kwargs.get("scale")
        for i, template in enumerate(templates):
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


            temp_path = os.path.join(TEMP_PATH,template)
            self.overlay_template_on_frame(temp_path,margin2,scale)
            margin2 = margin2 +100+h


        if kwargs.get("multiple"):
            for i in range(len(templates)):
                loc = all_loc[i]
                for pt in zip(*loc[::-1]):
                    cv2.rectangle(self.frame, pt, (pt[0] + w, pt[1] + h), colors[i], 2)

        self.put_text(kwargs.get("text"))  

    # Apply optical flow to the frames
    def optical_flow_FB(self,start_time,duration,**kwargs):
        end_time = start_time + duration - 1
        if not start_time <= self.current_time <= end_time:
            return
        prev_gray = cv2.cvtColor(self.previous_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(self.frame,cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prev_gray,curr_gray,None, 0.5, 3, 15, 3, 5, 1.2, 0) #is array in y,x

        step = 100
        scale = 2
        for y in range(0,curr_gray.shape[0],step):
            for x in range(0,curr_gray.shape[1],step):
                fx,fy = flow[y,x]
                cv2.arrowedLine(self.frame, (x, y), (int(x + fx*scale), int(y + fy*scale)), (0, 0, 255), 3, tipLength=0.3)
        self.put_text(kwargs.get("text"))

    # Detect keypoints for ORB
    def detect_kp_temp(self,template_string):        
        TEMP_PATH = os.path.join(self.CURRENT_PATH, template_string)
        templates = os.listdir(TEMP_PATH)
        im_temp = cv2.imread(os.path.join(TEMP_PATH,templates[0]))
        self.im_temp_gray = cv2.cvtColor(im_temp,cv2.COLOR_BGR2GRAY)
        self.orb = cv2.ORB_create()
        self.kp1,self.des1 = self.orb.detectAndCompute(self.im_temp_gray, None)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

    # Apply ORB and add dive stats to the bottle
    def orb_diver(self,start_time,duration,**kwargs):
        end_time = start_time + duration - 1
        if not start_time <= self.current_time <= end_time:
            return
        
        if self.orb is None:
            template_string = kwargs.get("templates")
            self.detect_kp_temp(template_string)

        frame_gray = cv2.cvtColor(self.frame,cv2.COLOR_BGR2GRAY)
        kp2, des2 = self.orb.detectAndCompute(frame_gray, None)
        matches = self.bf.match(self.des1,des2)
        sorted_matches = sorted(matches, key=lambda x: x.distance)
        good = sorted_matches[:50]
        MIN_MATCH_COUNT = 20
        if len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32([ self.kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            if H is not None:
                h, w = self.im_temp_gray.shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, H)

                new_center = np.mean(dst[:, 0, :], axis=0)  

                alpha = 0.9  
                if self.smooth_center is None:
                    self.smooth_center = new_center
                else:
                    self.smooth_center = alpha * self.smooth_center + (1 - alpha) * new_center

                x, y = self.smooth_center  
                info_text = f"Air: {150 :.0f} bar"

                cv2.putText(self.frame, info_text, (int(x)-100, int(y)-30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                
                if self.frame_number%(3*self.fps) ==0:
                    self.depth = random.uniform(7,8)
                depth_text = f"Depth: {self.depth:.1f} m"
                cv2.putText(self.frame, depth_text, (int(x)-100, int(y)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        self.frame_number+=1
        self.put_text(kwargs.get("text"))


    # run all the funtions
    def run(self, show_video=False):
        #list of exercises(fucntion, duration)
        exercises = [
            (self.temp_match, 20000,{"templates":"Templates_numbers","multiple":True,"text":"Template matching of student numbers","scale":4}),
            (self.temp_match, 5000,{"templates":"Templates_logo","text":"Template matching of student UT logo","scale":1}),
            (self.temp_match, 5000,{"templates":"Templates_photo","text":"Template matching of photo","scale":0.5}),
            (self.temp_match, 5000,{"templates":"Templates_laptop","text":"Template matching of laptop","scale":0.2}),
            (self.optical_flow_FB,5000,{"text":"Optical flow"}),
            (self.orb_diver,20000,{"templates":"sift_diver","text": "ORB feature detection of bottle"})            
        ]

        while self.cap.isOpened():
            ret, self.frame = self.cap.read()
            if not ret:
                break
            frame_copy = self.frame.copy()
            self.current_time = int(self.cap.get(cv2.CAP_PROP_POS_MSEC))
            self.downsample()

            start = 0


            for func, dur,kwargs in exercises:
                result = func(start, dur,**kwargs)
                start += dur
                
            
            self.out.write(self.frame)
            self.previous_frame = frame_copy

            if show_video:
                cv2.imshow('Video', self.frame)
            # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
