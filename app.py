from streamlit_webrtc import webrtc_streamer
import streamlit as st
import time
import av
import queue
from utils.ShoulderP import ShoulderP
from utils.Squat import Squat
from utils.BenchP import BenchP
from utils.DeadL import DeadL
from utils.my_helpers import StandardProcess, print_count, workout_row
from utils.KnnClassif import EMADictSmoothing
from utils.Drawing import drawing
import random
import numpy as np
import pandas as pd

frame = 15
rn = round(random.random(),5)
class VideoProcessor:
    def __init__(self):
        self.result_queue = queue.Queue()
        self.locked = False
        self.pose_args = 0
        self.prev_pose_frame = 'resting'
        self.pose_state = 'resting'
        self.workout = 'resting'

        #TODO: delte this and instaltiate them
        ShoulderP.times = 0
        DeadL.times = 0
        Squat.times = 0
        BenchP.times = 0


        self.set_no = 0
        self.r_time = 0
        self.w_time = 0
        self.table = pd.DataFrame()
        self.resting_time = 0
        self.workout_time = 0

        #debug settings
        self.debug = False
        self.goal = 0
        self.mod_comp = 2
        self.rest_thresh = 5
        self.ien = 5.3
        self.iex = 3.2
        self.iw = 60
        self.ia = 0.1
        self.len = 8.5
        self.lex = 3.5
        self.lw = 30
        self.la = 0.3
        self.top_n_mean = 10
        self.top_n_max = 30
        self.font_color = (255,255,255)
        self.vis_thresh = 0.7


        self.min_det_conf = 0.6
        self.min_trk_conf = 0.8
        self.Stdp = StandardProcess(
            model_complexity=self.mod_comp,
            min_detection_confidence=self.min_det_conf,
            min_tracking_confidence=self.min_trk_conf)
        self.smoother = EMADictSmoothing('utils/fitness_poses_csvs_out')




    def recv(self, frame):
        start = time.time()
        frame, landmarks, height, width = self.Stdp.std_process(frame, width= None, height= None)
        vis_sum = 0
        if landmarks is not None:
            for ld in landmarks.landmark:
                vis_sum += ld.visibility
            vis_av = vis_sum/33
            # plot sticks if debug
            frame = drawing.draw_landmarks(frame, landmarks, visibility_th=0.0) if self.debug else frame
            landmarks = None if vis_av < self.vis_thresh else landmarks

        if landmarks is not None:
            if not self.locked:
                pose_knn = self.Stdp.pose_class(landmarks, self.top_n_mean, self.top_n_max)
                self.pose_predict = self.smoother(pose_knn)
                self.smoother.set_rate(self.iw,self.ia)

                frame, _ = ShoulderP.run_sp(frame, self.pose_predict, landmarks, self.locked)
                frame, _ = Squat.run_sq(frame,self.pose_predict, landmarks, self.locked)
                Squat.set_thresh(self.ien, self.iex)
                frame, _ = BenchP.run_bp(frame, self.pose_predict, landmarks, self.locked)
                BenchP.set_thresh(self.ien, self.iex)
                frame, _ = DeadL.run_dl(frame, self.pose_predict, landmarks, self.locked)
                DeadL.set_thresh(self.ien, self.iex)
                drawing.annotation(frame, landmarks)


            else:
                #locked
                if self.pose_state == 'shoulder':
                    frame, l_pp = ShoulderP.run_sp(frame, self.pose_predict, landmarks, self.locked)
                    drawing.draw_lines(frame, landmarks, self.pose_state)
                    ShoulderP.set_param(self.len,self.lex,self.lw,self.la)
                elif self.pose_state == 'squat':
                    frame, l_pp = Squat.run_sq(frame, self.pose_predict, landmarks, self.locked)
                    drawing.draw_lines(frame, landmarks,self.pose_state)
                    Squat.set_param(self.len, self.lex, self.lw, self.la)
                elif self.pose_state == 'bench':
                    frame, l_pp = BenchP.run_bp(frame, self.pose_predict, landmarks, self.locked)
                    drawing.draw_lines(frame, landmarks, self.pose_state)
                    BenchP.set_param(self.len, self.lex, self.lw, self.la)
                elif self.pose_state == 'deadlift':
                    frame, l_pp = DeadL.run_dl(frame, self.pose_predict, landmarks, self.locked)
                    drawing.draw_lines(frame, landmarks, self.pose_state)
                    DeadL.set_param(self.len, self.lex, self.lw, self.la)
                self.pose_predict = self.smoother(l_pp)

        else:
            self.pose_predict = self.smoother({'resting':10})

        pose_frame = max(self.pose_predict, key=self.pose_predict.get)
        counts = [ShoulderP.times, Squat.times, BenchP.times, DeadL.times]
        count = np.max(counts)

        #?????? ???????????? resting??? ??????
        if pose_frame == 'resting':
            #????????? resting??? ????????? ?????? ???????????? resting??????
            if self.prev_pose_frame != 'resting':
                #interval ????????? ??? ????????? ?????? pose_state??? ?????? ?????????
                self.r_time = time.time()
            #????????? interval ????????? thresh??? ????????????
            if time.time() - self.r_time >self.rest_thresh:
                #??????????????????. ???????????? ?????? ?????? ?????? ??????.
                if self.pose_state != 'resting':
                    self.workout_time = time.time() - self.w_time
                    self.workout = self.pose_state
                self.pose_state = 'resting'
                self.locked = False
            #interval ????????? thresh ??????????????? ?????? state working out
            elif np.max(counts):
                if self.pose_state != 'resting':
                    self.pose_state = ['shoulder', 'squat', 'bench', 'deadlift'][np.argmax(counts)]
                    self.locked = True



        #?????? ???????????? ???????????? ??????
        else:
            # ?????? ???????????? ??????????????? pose_state??? 'resting'??? ?????? => ?????? ?????????
            if self.pose_state == 'resting':
                #?????? ????????? ?????? ????????????, ?????? ?????? ??????. pose_state??? ??????????????? ??????
                self.w_time = time.time()

                #skip initialization
                if self.r_time != 0:
                    self.resting_time = time.time() - self.r_time
                #???????????? ??????????????? pose_state??? ???????????? ??????
                if self.workout != 'resting' and count!=0:
                    self.set_no+=1
                    row = workout_row(set_no=self.set_no, pose=self.workout, count=count,
                                      set_duration=round(self.workout_time,2), rest_duration=round(self.resting_time-self.rest_thresh,2))
                    self.table = self.table.append(row)

                count = 0
                ShoulderP.times = 0
                DeadL.times = 0
                Squat.times = 0
                BenchP.times =0

                self.pose_state = pose_frame
            else:
                #??? ????????? ?????????, ??? pose_state ?????????
                if np.max(counts):
                    if self.pose_state != 'resting':
                        self.pose_state = ['shoulder', 'squat', 'bench', 'deadlift'][np.argmax(counts)]
                        self.locked = True
                    else:
                        self.locked = False
                else:
                    self.pose_state = pose_frame
                    self.locked = False
        count = np.max(counts)
        self.prev_pose_frame = pose_frame

        print(self.pose_predict)
        # print('takes', time.time()-start)
        self.result_queue.put(self.table)
        pos = self.pose_state
        frame = print_count(frame, height, width,
                            count, self.goal,
                            str(pos), str(round(self.pose_predict[pos]*10)),
                            self.w_time, self.r_time,self.rest_thresh,
                            self.font_color,
                            self.debug)
        return av.VideoFrame.from_ndarray(frame, format="bgr24")



def run():

    st.set_page_config(page_title="???????????? ????????????")
    st.title('???????????? ????????????')

    st.caption("???????????? 4????????? - ????????? - ?????? - ??????????????? - ??????????????? - ??? ?????? ?????? ??? ?????????")

    with st.expander("Tutorial", expanded = False):
        st.write("""
             ??????????????? 4??? ??????????????? ?????? ?????? ?????? ????????????.\n
             ??? ?????? ????????? ???????????? ??? ???????????? :(\n
             ???????????? reps??? sets??? ???????????? ?????? ????????? ?????? ????????? ?????? ??? ???????????? ^0^\n
             ??? **???????????? ???????????????** ????????? ???????????? ??? ?????? ?????????\n
             ?????????????????? ??????????????? ???????????????, ????????? ????????? ??? ???????????? ??????????????? :)\n
             ????????? **?????????**??? ?????? ?????? ???????????? ????????? ???????????????!!             
            """)
        st.image("utils/cache/example.jpg")
        st.write("""???????????? ????????? ????????? ?????? ??? ????????????\n
                    ??????: ?????????, jeenie37@hanyang.ac.kr
                """)
        debug = st.checkbox('Debug Mode')
        if debug:
            mdl_cp = st.slider('model complexity', value=2, min_value=0, max_value=2)
            rest_thresh = st.slider('resting threshold', value = 5, min_value =1, max_value = 30)

            col1, col2 = st.columns(2)
            ins = col1.slider('initial enter sensitivity', value=5.3, min_value=float(0), max_value=float(10))
            ixs = col1.slider('initial exit sensitivity', value=3.2, min_value=float(0), max_value=float(10))
            iew = col1.slider('initial ema window', value=60, min_value=0, max_value=300)
            iea = col1.slider('initial ema alpha', value=0.1, min_value=float(0), max_value=float(1))

            lns = col2.slider('locked enter sensitivity', value=6.5, min_value=float(0), max_value=float(10))
            lxs = col2.slider('locked exit sensitivity', value=3.5, min_value=float(0), max_value=float(10))
            lew = col2.slider('locked ema window', value=30, min_value=0, max_value=100)
            lea = col2.slider('locked ema alpha', value=0.3, min_value=float(0), max_value=float(1))
            top_mean_n = col1.slider('top_n_mean', value = 50, min_value = 10, max_value = 100)
            top_max_n = col1.slider('top_n_max', value=70, min_value=10, max_value=100)
            vis_thresh = col2.slider('visualization threshold', value =0.7, min_value=float(0), max_value=float(1))

            f_color = st.color_picker('pick font color')

    goal = st.select_slider('How many?', [i for i in range(0, 21)])
    ctx = webrtc_streamer(key="example", video_processor_factory=VideoProcessor,
                    media_stream_constraints={"video": {"frameRate": {"ideal": frame}}})

    labels_placeholder = st.empty()

    if ctx.video_processor and debug:
        ctx.video_processor.goal = goal
        ctx.video_processor.debug = debug
        ctx.video_processor.rest_thresh = rest_thresh
        ctx.video_processor.ien = ins
        ctx.video_processor.iex = ixs
        ctx.video_processor.iw = iew
        ctx.video_processor.ia= iea
        ctx.video_processor.len = lns
        ctx.video_processor.lex = lxs
        ctx.video_processor.lw = lew
        ctx.video_processor.la = lea
        ctx.video_processor.mod_comp = mdl_cp
        ctx.video_processor.top_n_mean = top_mean_n
        ctx.video_processor.top_n_max = top_max_n
        f_color = f_color.lstrip('#')
        f_color = tuple(int(f_color[i:i + 2], 16) for i in (4, 2, 0))
        ctx.video_processor.vis_thresh = vis_thresh
        ctx.video_processor.font_color = f_color







    result = pd.DataFrame()
    result.to_pickle('utils/cache/results/result{}.pkl'.format(rn))
    if ctx.state.playing:
        while True:
            if ctx.video_processor:
                try:
                    result = ctx.video_processor.result_queue.get(
                        timeout=1.0
                    )
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
