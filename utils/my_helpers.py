from utils.KnnClassif import FullBodyPoseEmbedder, PoseClassifier, EMADictSmoothing
import mediapipe as mp
import cv2
import numpy as np

class StandardProcess:

    def __init__(self, model_complexity, av_size, av_alpha):
        self.pose = mp.solutions.pose.Pose(model_complexity=model_complexity)
        self.pose_embedder = FullBodyPoseEmbedder()
        self.pose_classifier = PoseClassifier(
            pose_samples_folder='utils/fitness_poses_csvs_out',
            pose_embedder=self.pose_embedder,
            top_n_by_max_distance=30,
            top_n_by_mean_distance=10)
        self.pose_classification_filter = EMADictSmoothing(
            window_size=av_size,
            alpha=av_alpha)


    def std_process(self, frame, width = None, height = None):
        # (480 640 3)
        frame = frame.to_ndarray(width= width, height = height, format="bgr24")
        # batch인데 어차피 1임
        frame = cv2.flip(frame,1)
        results = self.pose.process(frame)
        landmarks = results.pose_landmarks

        self.frame_height, self.frame_width, _ = frame.shape

        return frame, landmarks

    def pose_class(self, landmarks):
        landmarks_np = np.array([[lmk.x * self.frame_width, lmk.y * self.frame_height, lmk.z * self.frame_width]
                                 for lmk in landmarks.landmark], dtype=np.float32)
        pose_classification = self.pose_classifier(landmarks_np)
        averaged_classification = self.pose_classification_filter(pose_classification)

        return averaged_classification
