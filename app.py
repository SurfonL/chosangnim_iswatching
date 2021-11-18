from streamlit_webrtc import webrtc_streamer, WebRtcMode
import streamlit as st

import cv2
import time
import av
from utils.ShoulderP import ShoulderP
from utils.my_helpers import StandardProcess
import numpy as np
frame = 15


#TODO: test av_size, av_alpha, framewidth+height
class VideoProcessor:
    def __init__(self):

        self.Stdp = StandardProcess(
            model_complexity = 0,
            av_size = 60,
            av_alpha = 0.4
        )
        self.count = 0
        self.goal = 0
        self.mode = ""



    def recv(self, frame):
        start = time.time()
        frame, landmarks = self.Stdp.std_process(frame, width= None, height= None)
        if landmarks is not None:
            pose_predict = self.Stdp.pose_class(landmarks)
            pose_state = max(pose_predict,key=pose_predict.get)
            print(pose_state)

            #TODO: record workout, rest time
            #if pos state = shoulderp_down
            frame, self.count = ShoulderP.run_shoulderp(frame,landmarks)
                    #also draw what pose it is
            #if pos_stae = squat_down
                #frame = Squat.run_squat
                    # also draw what pose it is
            #if pos_state = bench_ up or bench down
                # frame = BenchP.run_benchp(pos_state, landmarks)
                #     pos state down then up => count
                #     draw
            #if pose_state = dead_up or bench down
                #   frame = Dead.run_dead(pos_state, landmarks)
                #     pos state down then up => count
                #     draw

        text = "Counts : " + str(self.count)
        frame = cv2.putText(frame, text, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)



        # print('takes', time.time()-start)
        return av.VideoFrame.from_ndarray(frame, format="bgr24")



def run():
    st.title('조상님이 보고있다')
    #subtitle(4대운동 - 스쿼트 - 벤치 - 데드리프트 - 숄더프레스 보조)
    # mode = st.sidebar.selectbox("", ['Shoulder Press', "Squats"])
    ctx = webrtc_streamer(key="example", video_processor_factory=VideoProcessor,
                    media_stream_constraints={"video": {"frameRate": {"ideal": frame}}})
    goal = st.select_slider('How many?', [i for i in range(1, 21)])
    if ctx.video_processor:
        ctx.video_processor.goal = goal
        ctx.video_processor.frame = frame

    #add columns - workout time, count, rest time
    #reset column
    #delete columns

run()

#todo: statistic table
#record sets and time automatically
#reset button
#자세 교정
#집중물어