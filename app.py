from streamlit_webrtc import webrtc_streamer, WebRtcMode
import streamlit as st
import mediapipe as mp
import time
import av
from utils.ShoulderP import ShoulderP
from utils import my_helpers

from utils.KnnClassif import FullBodyPoseEmbedder, PoseClassifier, EMADictSmoothing


class VideoProcessor:
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(model_complexity=2)
        # self.State = my_helpers.PoseState(buffer_size)

        self.goal = 0
        self.mode = ""

        # self.pose_embedder = FullBodyPoseEmbedder()
        # self.pose_classifier = PoseClassifier(
        #     pose_samples_folder='utils/fitness_poses_csvs_out',
        #     pose_embedder=self.pose_embedder,
        #     top_n_by_max_distance=30,
        #     top_n_by_mean_distance=10)
        # self.pose_classification_filter = EMADictSmoothing(
        #     window_size=10,
        #     alpha=0.2)

    def recv(self, frame):
        start = time.time()
        frame, landmarks = my_helpers.std_process(frame,self.pose, width= None, height= None)
        if landmarks is not None:
            #frame_pos = knn classifier

            # pose_classification = self.pose_classifier(landmarks)
                #append to deque -> most_common -> return most common state
            # pose_classification_filtered = self.pose_classification_filter(pose_classification)


            #TODO: record workout, rest time

            #if pos state = shoulderp_down
            frame = ShoulderP.run_shoulderp(frame,landmarks)
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



        # print('takes', time.time()-start)
        return av.VideoFrame.from_ndarray(frame, format="bgr24")



def run():
    st.title('조상님이 보고있다')
    #subtitle(4대운동 - 스쿼트 - 벤치 - 데드리프트 - 숄더프레스 보조)
    mode = st.sidebar.selectbox("", ['Shoulder Press', "Squats"])
    ctx = webrtc_streamer(key="example", video_processor_factory=VideoProcessor,
                    media_stream_constraints={"video": {"frameRate": {"ideal": 10}}})
    goal = st.select_slider('How many?', [i for i in range(1, 21)])
    if ctx.video_processor:
        ctx.video_processor.goal = goal
        ctx.video_processor.mode = mode

    #add columns - workout time, count, rest time
    #reset column
    #delete columns

run()

#todo: statistic table
#record sets and time automatically
#reset button
#자세 교정
#집중물어