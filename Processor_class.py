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

        print('Processing new video')


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

    # Make the frames gray
    # def to_gray(self):        
    #     if self.frame.shape[2] == 1:
    #         return
    #     else:
    #         self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
    #         # self.frame = cv2.cvtColor(self.frame, cv2.COLOR_GRAY2BGR) #for the text


    def temp_match(self,start_time,duration):
        end_time = start_time + duration - 1
        if not start_time <= self.current_time <= end_time:
            return
        CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
        TEMP_PATH = os.path.join(CURRENT_PATH,'Templates_4')
        templates = os.listdir(TEMP_PATH)
        frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        # for template in templates:
        template = templates[0]
        temp_im = cv2.imread(os.path.join(TEMP_PATH,template),cv2.IMREAD_GRAYSCALE)
        h,w = temp_im.shape 
        res = cv2.matchTemplate(frame_gray,temp_im,cv2.TM_CCOEFF_NORMED)
    # plt.figure()
    # plt.imshow(res)
    # plt.axis('off')
    # plt.show()
        threshold = 0.75
        loc = np.where( res >= threshold)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(self.frame, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
        # print(f"{template}: {len(loc[0])} matches gevonden")
            
        # cv2.imwrite('res.png',img)


   

    # run all the funtions
    def run(self, show_video=False):
        #list of filters (funciton,duration(ms))
        exercises = [
            # (self.show_template, 6000),
            (self.temp_match, 20000),
        ]

        while self.cap.isOpened():
            ret, self.frame = self.cap.read()
            if not ret:
                break
            self.current_time = int(self.cap.get(cv2.CAP_PROP_POS_MSEC))
            self.downsample()

            start = 0


            for func, dur in exercises:
                result = func(start, dur)
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

        self.downsample()
        

    
        exercises = [
            # (self.show_template, 6000),
            (self.temp_match, 20000),
        ]

        start = 0

        for func, dur in exercises:
            result = func(start, dur)
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
 
