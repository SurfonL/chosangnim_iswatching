import pandas as pd
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import streamlit as st
import cv2
import time
import av
import queue
from utils.ShoulderP import ShoulderP
from utils.my_helpers import StandardProcess, print_count, workout_row
import numpy as np
frame = 15


#TODO: test av_size, av_alpha, framewidth+height
class VideoProcessor:
    def __init__(self):

        self.Stdp = StandardProcess(
            model_complexity = 0,
            av_size = 100,
            av_alpha = 0.01
        )
        self.result_queue = queue.Queue()
        self.prev_pose_state = None
        self.table = pd.DataFrame()
        self.count = 0
        self.goal = 0


    def recv(self, frame):
        start = time.time()
        frame, landmarks, height, width = self.Stdp.std_process(frame, width= None, height= None)
        if landmarks is not None:
            pose_predict = self.Stdp.pose_class(landmarks)
            pose_state = max(pose_predict,key=pose_predict.get)
            pose_prob = pose_predict[pose_state]
            #if pose_prob < 6 then resting

            #TODO: record workout, rest time
            #if pos state = shoulderp_down
            frame, self.count = ShoulderP.run_shoulderp(frame,landmarks)
            pos = 'shoulder_p'
            if self.prev_pose_state == 'resting':
                ShoulderP.times = 0
                self.count = 0
                self.set_time = time.time()

            # if pos_stae = squat_down
            #     frame = Squat.run_squat
            #         also draw what pose it is
            # if pos_state = bench_ up or bench down
            #     frame = BenchP.run_benchp(pos_state, landmarks)
            #         pos state down then up => count
            #         draw
            # if pose_state = dead_up or bench down
            #       frame = Dead.run_dead(pos_state, landmarks)
            #         pos state down then up => count
            #         draw
            #
            # if pose_state == 'resting' and self.prev_pose_state != 'resting':
            #     pass



            self.prev_pose_state = pose_state
        else:
            pose_state = None


        # print('takes', time.time()-start)
        row = workout_row(set_no=1, pose=pose_state, count=self.count, set_duration=30, rest_duration=40)
        self.table = row
        self.result_queue.put(self.table)
        pos = pose_state
        frame = print_count(frame, height, width, self.count, self.goal, str(pos))
        return av.VideoFrame.from_ndarray(frame, format="bgr24")



def run():
    st.title('조상님이 보고있다')
    #subtitle(4대운동 - 스쿼트 - 벤치 - 데드리프트 - 숄더프레스 보조)
    ctx = webrtc_streamer(key="example", video_processor_factory=VideoProcessor,
                    media_stream_constraints={"video": {"frameRate": {"ideal": frame}}})
    goal = st.select_slider('How many?', [i for i in range(0, 21)])
    if ctx.video_processor:
        ctx.video_processor.goal = goal
        value = ctx.video_processor.count

    if ctx.state.playing:
        labels_placeholder = st.empty()
        # NOTE: The video transformation with object detection and
        # this loop displaying the result labels are running
        # in different threads asynchronously.
        # Then the rendered video frames and the labels displayed here
        # are not strictly synchronized.
        while True:
            if ctx.video_processor:
                try:
                    result = ctx.video_processor.result_queue.get(
                        timeout=1.0
                    )
                except queue.Empty:
                    result = None
                labels_placeholder.table(result)
            else:
                break

    #add columns - workout time, count, rest time
    #reset column
    #delete columns

run()

#todo: statistic table
#record sets and time automatically
#reset button
#자세 교정
#집중물어