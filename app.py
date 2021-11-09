from streamlit_webrtc import webrtc_streamer
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

    def recv(self, frame):
        start = time.time()

        frame, frame_coordinates, frame_height,frame_width = \
            my_helpers.std_process(frame,self.model_variant, self.model, self.resolution,self.framework)

        frame = ShoulderP.run_shoulderp(frame, frame_coordinates, frame_height, frame_width)

        print(self.goal)
        # print('takes', time.time()-start)
        return av.VideoFrame.from_ndarray(frame, format="bgr24")

def run():
    print('this also runs')
    st.title('조상님이 보고있다')
    st.selectbox("", ['Shoulder Press', "Squats"])
    ctx = webrtc_streamer(key="example", video_processor_factory=VideoProcessor,
                    media_stream_constraints={"video": {"frameRate": {"ideal": 20}}})
    goal = st.select_slider('how many', [i for i in range(1, 21)])
    if ctx.video_processor:
        ctx.video_processor.goal = goal


run()