from streamlit_webrtc import webrtc_streamer, WebRtcMode

import av
from utils.ShoulderP import ShoulderP
from utils import my_helpers
import time
import streamlit as st

import mediapipe as mp



class VideoProcessor:
    def __init__(self):
        self.pose = mp.solutions.pose.Pose()

        self.goal = 0
        self.mode = ""

    def recv(self, frame):
        start = time.time()
        frame, landmarks = my_helpers.std_process(frame,self.pose)
        if landmarks is not None:
            frame = ShoulderP.run_shoulderp(frame,landmarks)



        # print('takes', time.time()-start)
        return av.VideoFrame.from_ndarray(frame, format="bgr24")



def run():
    st.title('조상님이 보고있다')
    mode = st.sidebar.selectbox("", ['Shoulder Press', "Squats"])
    ctx = webrtc_streamer(key="example", video_processor_factory=VideoProcessor,
                    media_stream_constraints={"video": {"frameRate": {"ideal": 10}}})
    goal = st.select_slider('How many?', [i for i in range(1, 21)])
    if ctx.video_processor:
        ctx.video_processor.goal = goal
        ctx.video_processor.mode = mode

run()

#todo: statistic table
#record sets and time automatically
#reset button
#자세 교정
#집중물어