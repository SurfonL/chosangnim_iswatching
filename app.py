import pandas as pd
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import streamlit as st
import time
import av
import queue
from utils.ShoulderP import ShoulderP
from utils.Squat import Squat
from utils.BenchP import BenchP
from utils.DeadL import DeadL
from utils.my_helpers import StandardProcess, print_count, workout_row, draw_landmarks
from utils.KnnClassif import EMADictSmoothing
import random
import numpy as np

frame = 15
rest_thresh = 5
rn = round(random.random(),5)


#TODO: test av_size, av_alpha, framewidth+height
class VideoProcessor:
    def __init__(self):
        self.Stdp = StandardProcess(
            model_complexity = 0,
        )
        self.smoother = EMADictSmoothing(
            window_size=60,
            alpha=0.1)

        self.result_queue = queue.Queue()
        self.locked = False
        self.pose_args = 0
        self.prev_pose_frame = 'resting'
        self.pose_state = 'resting'
        self.workout = 'resting'

        self.count = 0
        self.count_rec = 0
        self.set_no = 0
        self.goal = 0

        self.r_time = 0
        self.w_time = 0
        self.table = pd.DataFrame()
        self.resting_time = 0
        self.workout_time = 0

        self.debug = False



    def recv(self, frame):
        start = time.time()
        frame, landmarks, height, width = self.Stdp.std_process(frame, width= None, height= None)
        if landmarks is not None:
            frame = frame
            pose_knn = self.Stdp.pose_class(landmarks)
            pose_predict = self.smoother(pose_knn)
            pose_frame = max(pose_predict,key=pose_predict.get)
            pose_prob = pose_predict[pose_frame]

            if not self.locked:
                self.smoother.set_rate(60,0.1)
                en = 5
                ex = 6
                if pose_frame == 'shoulder':
                    frame, self.count = ShoulderP.run_shoulderp(frame,landmarks)
                elif pose_frame == 'squat_down':
                    frame, self.count = Squat.run_sq(frame,pose_predict)
                    Squat.set_thresh(en,ex)
                elif pose_frame == 'bench_down':
                    frame, self.count = BenchP.run_bp(frame, pose_predict)
                    BenchP.set_thresh(en,ex)
                elif pose_frame == 'dead_down':
                    frame, self.count = DeadL.run_dl(frame, pose_predict)
                    DeadL.set_thresh(en, ex)

            else:
                #locked
                self.smoother.set_rate(10, 0.2)
                if self.pose_state == 'shoulder':
                    frame, self.count = ShoulderP.run_shoulderp(frame, landmarks)
                elif self.pose_state == 'squat_down':
                    frame, self.count = Squat.run_sq(frame, pose_predict)
                elif self.pose_state == 'bench_down':
                    frame, self.count = BenchP.run_bp(frame, pose_predict)
                elif self.pose_state == 'dead_down':
                    frame, self.count = DeadL.run_dl(frame, pose_predict)


            frame = draw_landmarks(frame, landmarks, visibility_th=0.3) if self.debug else frame

        else:
            pose_predict = self.smoother({'resting':10})
            pose_frame = max(pose_predict, key=pose_predict.get)

        counts = [ShoulderP.times, Squat.times, BenchP.times, DeadL.times]

        #현재 프레임이 resting인 경우
        if pose_frame == 'resting':
            #직전이 resting이 아닌데 현재 프레임이 resting이면
            if self.prev_pose_frame != 'resting':
                #interval 시간을 잼 그리고 일단 pose_state는 하던 운동임
                self.r_time = time.time()
                # self.pose_state = self.prev_pose_frame
            #그런데 interval 시간이 15초 이상이면
            if time.time() - self.r_time >rest_thresh:
                #휴식시간이다. 지금까지 세트 운동 시간 기록.
                if self.pose_state != 'resting':
                    self.workout_time = time.time() - self.w_time
                    self.workout = self.pose_state
                self.pose_state = 'resting'


        #현재 프레임이 운동중인 경우
        else:
            # 현재 프레임이 운동중이고 pose_state가 'resting'인 경우 => 운동 시작함
            if self.pose_state == 'resting':
                #운동 시간을 재기 시작하고, 휴식 시간 기록. pose_state를 운동중으로 바꿈
                self.w_time = time.time()

                #skip initialization
                if self.r_time != 0:
                    self.resting_time = time.time() - self.r_time
                #프레임이 운동중인데 pose_state도 운동중인 경우
                if self.workout != 'resting' and self.count!=0:
                    self.set_no+=1
                    row = workout_row(set_no=self.set_no, pose=self.workout, count=self.count, set_duration=round(self.workout_time,2),
                                      rest_duration=round(self.resting_time,2))
                    self.table = self.table.append(row)

                self.count = 0
                ShoulderP.times = 0
                DeadL.times = 0
                Squat.times = 0
                BenchP.times =0

                self.pose_state = pose_frame
            else:
                #현 프레임 운동중, 현 pose_state 운동중
                if np.max(counts):
                    self.pose_state = ['shoulder', 'squat_down', 'bench_down', 'dead_down'][np.argmax(counts)]
                    self.locked = True
                else:
                    self.pose_state = pose_frame
                    self.locked = False

        self.prev_pose_frame = pose_frame



        # print('takes', time.time()-start)
        self.result_queue.put(self.table)
        pos = self.pose_state
        #TODO: draw time, squat_down -> squat
        frame = print_count(frame, height, width,
                            self.count, self.goal,
                            str(pos), str(round(pose_predict[pos]*10)),
                            self.w_time, self.r_time,
                            self.debug)
        return av.VideoFrame.from_ndarray(frame, format="bgr24")



def run():

    st.set_page_config(page_title="조상님이 지켜본다")
    st.title('조상님이 지켜본다')

    st.caption("조상님이 4대운동 - 스쿼트 - 벤치 - 데드리프트 - 숄더프레스 - 를 대신 기록 해 줍니다")

    with st.expander("Tutorial", expanded = False):
        st.write("""
             헬스장에서 4대 중량운동을 돕기 위해 만든 앱입니다.\n
             그 외의 운동은 도와드릴 수 없습니다 :(\n
             자동으로 reps와 sets를 세어주고 운동 시간과 휴식 시간을 기록 해 준답니다 ^0^\n
             꼭 **머리부터 발가락까지** 나오게 카메라를 잘 놓아 주세요\n
             전체화면으로 사용하기를 권장드리며, 기록된 운동은 앱 하단부를 확인하세요 :)\n
             그리고 **데이터**를 정말 많이 사용하기 때문에 조심하세요!!             
            """)
        st.image("utils/cache/example.jpg")
        st.write("""사용자가 많으면 에러가 있을 수 있습니다\n
                    문의: 전우진, jeenie37@hanyang.ac.kr
                """)
        debug = st.checkbox('Debug Mode')
        if debug:
            mdl_cp = st.slider('model complexity', value=2, min_value=0, max_value=2)
            rest_thresh = st.slider('resting threshold', value = 5, min_value =1, max_value = 30)

            col1, col2 = st.columns(2)
            ins = col1.slider('initial enter sensitivity', value=5.0, min_value=float(0), max_value=float(10))
            ixs = col1.slider('initial exit sensitivity', value=6.0, min_value=float(0), max_value=float(10))
            iew = col1.slider('initial ema window', value=60, min_value=0, max_value=300)
            iea = col1.slider('initial ema alpha', value=0.1, min_value=float(0), max_value=float(1))

            lns = col2.slider('locked enter sensitivity', value=5.0, min_value=float(0), max_value=float(10))
            lxs = col2.slider('locked exit sensitivity', value=6.0, min_value=float(0), max_value=float(10))
            lew = col2.slider('locked ema window', value=10, min_value=0, max_value=100)
            lea = col2.slider('locked ema alpha', value=0.2, min_value=float(0), max_value=float(1))


    goal = st.select_slider('How many?', [i for i in range(0, 21)])
    ctx = webrtc_streamer(key="example", video_processor_factory=VideoProcessor,
                    media_stream_constraints={"video": {"frameRate": {"ideal": frame}}})

    labels_placeholder = st.empty()

    if ctx.video_processor:
        ctx.video_processor.goal = goal
        ctx.video_processor.debug = debug


    result = pd.DataFrame()
    result.to_pickle('utils/cache/results/result{}.pkl'.format(rn))
    if ctx.state.playing:
        while True:
            if ctx.video_processor:
                try:
                    result = ctx.video_processor.result_queue.get(
                        timeout=1.0
                    )
                    #TODO: 사람마다 다르게 해야함
                    result.to_pickle('utils/cache/results/result{}.pkl'.format(rn))
                except queue.Empty:
                    result = None
                labels_placeholder.table(result)
            else:

                break
    else:
        result = pd.read_pickle('utils/cache/results/result{}.pkl'.format(rn))
        labels_placeholder.table(result)








run()
