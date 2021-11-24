from utils.Workouts import Workouts
from utils.Drawing import drawing

class BenchP(Workouts):
    def __init__(self):
        super().__init__()
        self._class_name = 'bench_down'
        self._pose_samples_folder = 'utils/pose_plots/bench'

    def run_bp(self,frame, landmarks):
        pose_knn = self.pose_classifier(landmarks)
        pose_predict = self.smoother(pose_knn)
        self.count(pose_predict)
        #draw things
        #frame = draw_bp(frame)

        return pose_predict


