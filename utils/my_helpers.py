import pandas as pd
import time

from utils.KnnClassif import FullBodyPoseEmbedder, PoseClassifier, EMADictSmoothing
import mediapipe as mp
import cv2
import numpy as np

class StandardProcess:

    def __init__(self, model_complexity, min_detection_confidence=0.5,
            min_tracking_confidence=0.5):
        self.pose = mp.solutions.pose.Pose(model_complexity=model_complexity,
                                           min_detection_confidence=min_detection_confidence,
                                           min_tracking_confidence=min_tracking_confidence)
        self.pose_embedder = FullBodyPoseEmbedder()
        self.pose_classifier = PoseClassifier(
            pose_samples_folder='utils/fitness_poses_csvs_out',
            pose_embedder=self.pose_embedder,
            top_n_by_max_distance=30,
            top_n_by_mean_distance=10)


    def std_process(self, frame, width = None, height = None):
        # (480 640 3)
        frame = frame.to_ndarray(width= width, height = height, format="bgr24")
        # batch인데 어차피 1임
        frame = cv2.flip(frame,1)
        results = self.pose.process(frame)
        landmarks = results.pose_landmarks

        self.frame_height, self.frame_width, _ = frame.shape
        return frame, landmarks, self.frame_height, self.frame_width

    def pose_class(self, landmarks, n_min, n_max):
        landmarks_np = np.array([[lmk.x * self.frame_width, lmk.y * self.frame_height, lmk.z * self.frame_width]
                                 for lmk in landmarks.landmark], dtype=np.float32)
        self.pose_classifier.set_minmaxn(n_min, n_max)
        pose_classification = self.pose_classifier(landmarks_np)

        return pose_classification

def print_count(frame,height,width,count, goal, pose, pose_prob, w_time, r_time,rest_thresh, font_color = (255,255,255), debug=True):

    if pose == 'bench':
        pose = 'bench press'

    if goal != 0:
        count = goal-count
        text = str(count) + " to go"
    else:
        text = "Count: " + str(count)
    f_size = height / 200
    f_thick = int(f_size * 1.5)
    t_size, t_y = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, f_size, f_thick)
    frame = cv2.putText(frame, text, (width - t_size[0] - int(width / 20), t_size[1] + int(height / 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, f_size, font_color, f_thick)

    f_size = height/400
    f_thick = f_thick -1
    t_size, t_y2 = cv2.getTextSize(pose, cv2.FONT_HERSHEY_SIMPLEX, f_size, f_thick)
    frame = cv2.putText(frame, pose, (width - t_size[0] - int(width / 20), t_size[1] + int(height / 15) +t_y*3),
                        cv2.FONT_HERSHEY_SIMPLEX, f_size, font_color, f_thick)

    now = time.time()
    if pose == 'resting':
        r_time = now if r_time < rest_thresh else r_time+rest_thresh
        t = str(round(now-r_time,1))
    else:
        t = str(round(now-w_time,1))
    f_size = height/400
    f_thick = f_thick -1
    t_size, _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, f_size, f_thick)
    frame = cv2.putText(frame, t, (width - t_size[0] - int(width / 20), t_size[1] + int(height / 15) +t_y*3+t_y2*4),
                        cv2.FONT_HERSHEY_SIMPLEX, f_size, font_color, f_thick)

    if debug:
        f_size = height / 600
        f_thick = f_thick - 1 if f_thick >1 else 1
        t_size, _ = cv2.getTextSize(pose_prob, cv2.FONT_HERSHEY_SIMPLEX, f_size, f_thick)
        frame = cv2.putText(frame, pose_prob,
                            (width - t_size[0] - int(width / 20), t_size[1] + int(height / 15) + t_y * 3 + t_y2*7),
                            cv2.FONT_HERSHEY_SIMPLEX, f_size, font_color, f_thick)

    return frame

def workout_row(set_no, pose, count, set_duration, rest_duration):
    columns = ['pose', 'count','set duration', 'rest duration']
    row = {'Set No: {}'.format(set_no): [pose, count, set_duration, rest_duration]}

    return pd.DataFrame.from_dict(row, orient='index', columns = columns)



