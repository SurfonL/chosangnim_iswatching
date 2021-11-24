from utils.Workouts import Workouts

class DeadL(Workouts):

    def __init__(self):

        self._class_name = 'dead_down'
        self.init('utils/pose_plots/deadlift')


    def run_dl(self, frame, landmarks, landmarks_np):
        pose_knn = self.pose_classifier(landmarks_np)
        pose_predict = self.smoother(pose_knn)
        self.count(pose_predict)

        return pose_predict



