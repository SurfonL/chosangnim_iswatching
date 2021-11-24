from utils.Workouts import Workouts

class DeadL(Workouts):

    def __init__(self):
        super().__init__()
        self._class_name = 'dead_down'
        self._pose_samples_folder = 'utils/pose_plots/deadlift'


    def run_dl(self, frame, landmarks):
        pose_knn = self.pose_classifier(landmarks)
        pose_predict = self.smoother(pose_knn)
        self.count(pose_predict)

        return pose_predict



