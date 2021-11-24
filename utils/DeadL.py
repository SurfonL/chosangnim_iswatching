from utils.Workouts import Workouts

class DeadL(Workouts):

    def __init__(self):

        self._class_name = 'dead_down'
        Workouts._pose_samples_folder = 'utils/pose_plots/deadlift'
        super().__init__()


    def run_dl(self, frame, landmarks, landmarks_np):
        pose_knn = self.pose_classifier(landmarks, landmarks_np)
        pose_predict = self.smoother(pose_knn)
        self.count(pose_predict)

        return pose_predict



