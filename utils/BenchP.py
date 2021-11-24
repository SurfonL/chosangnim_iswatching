from utils.Workouts import Workouts
from utils.Drawing import drawing

class BenchP(Workouts):
    def __init__(self):

        self._class_name = 'bench'
        self.init('utils/pose_plots/bench')


    def run_bp(self,frame, landmarks, landmarks_np):
        pose_knn = self.pose_classifier(landmarks_np)
        pose_predict = self.smoother(pose_knn)
        self.count(pose_predict)
        #draw things
        #frame = draw_bp(frame)

        return pose_predict


