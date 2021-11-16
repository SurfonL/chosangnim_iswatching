from streamlit_webrtc import webrtc_streamer, WebRtcMode

import av
from utils.ShoulderP import ShoulderP
from utils import my_helpers
import time
import streamlit as st



class VideoProcessor:
    def __init__(self):
        self.framework = 'tflite'
        self.model_variant = 'rt_lite'
        # self.framework = 'tensorflow'
        # self.model_variant = 'ii'
        self.model, self.resolution = my_helpers.get_model(self.framework, self.model_variant)

        self.goal = 0
        self.mode = ""

    def recv(self, frame):
        start = time.time()

        frame, frame_coordinates, frame_height,frame_width = \
            my_helpers.std_process(frame,self.model_variant, self.model, self.resolution,self.framework)

        print(frame.shape, 'after')

        if self.mode == 'Shoulder Press':
            frame = ShoulderP.run_shoulderp(frame, frame_coordinates, frame_height, frame_width)
        elif self.mode == "Squats":
            pass
        # print('takes', time.time()-start)
        return av.VideoFrame.from_ndarray(frame, format="bgr24")

def run():
    st.title('조상님이 보고있다')
    mode = st.sidebar.selectbox("", ['Shoulder Press', "Squats"])
    ctx = webrtc_streamer(key="example", video_processor_factory=VideoProcessor,
                    media_stream_constraints={"video": {"frameRate": {"ideal": 30}}})
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