from utils.Drawing import drawing
from KnnClassif import FullBodyPoseEmbedder,PoseClassifier, EMADictSmoothing
import numpy as np


class Squat:
    _class_name = 'squat_down'
    times = 0
    _pose_entered = False
    _enter_threshold = 6
    _exit_threshold = 4

    pose_embedder = FullBodyPoseEmbedder()
    pose_classifier = PoseClassifier(
        pose_samples_folder='utils/pose_plots/squat',
        pose_embedder=pose_embedder,
        top_n_by_max_distance=30,
        top_n_by_mean_distance=10)
    smoother = EMADictSmoothing('utils/pose_plots/squat')

    @classmethod
    def count(cls, pose_classification):
       """Counts number of repetitions happend until given frame.

       We use two thresholds. First you need to go above the higher one to enter
       the pose, and then you need to go below the lower one to exit it. Difference
       between the thresholds makes it stable to prediction jittering (which will
       cause wrong counts in case of having only one threshold).

       Args:
         pose_classification: Pose classification dictionary on current frame.
           Sample:
             {
               'pushups_down': 8.3,
               'pushups_up': 1.7,
             }

       Returns:
         Integer counter of repetitions.
       """
       # Get pose confidence.
       pose_confidence = 0.0
       if cls._class_name in pose_classification:
           pose_confidence = pose_classification[cls._class_name]

       # On the very first frame or if we were out of the pose, just check if we
       # entered it on this frame and update the state.
       if not cls._pose_entered:
           cls._pose_entered = pose_confidence > cls._enter_threshold
           return cls.times

       # If we were in the pose and are exiting it, then increase the counter and
       # update the state.
       if pose_confidence < cls._exit_threshold:
           cls.times += 1
           cls._pose_entered = False

    @classmethod
    def set_thresh(cls, enter, exit):
        cls._enter_threshold = enter
        cls._exit_threshold = exit


    @staticmethod
    def draw_circle(frame, landmarks):
        frame_height, frame_width = frame.shape[0], frame.shape[1]
        right_hip = landmarks.landmark[23]
        left_hip = landmarks.landmark[24]
        frame = drawing.image_alpha(frame, right_hip.x * frame_width, right_hip.y * frame_height, 30, (0, 255, 0), 0.3,
                                    1, 1)
        frame = drawing.image_alpha(frame, left_hip.x * frame_width, left_hip.y * frame_height, 30, (0, 255, 0), 0.3, 1,
                                    1)
        return frame

    @classmethod
    def run_sq(cls, frame, pose_predict, landmarks, locked=False):
        if locked:
            frame_height, frame_width = frame.shape[0], frame.shape[1]
            landmarks_np = np.array([[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                                     for lmk in landmarks.landmark], dtype=np.float32)
            pose_classification = cls.pose_classifier(landmarks_np)
            pose_predict = cls.smoother(pose_classification)

        cls.count(pose_predict)
        cls.draw_circle(frame, landmarks)
        # draw things
        # frame = draw_bp(frame)

        return frame

