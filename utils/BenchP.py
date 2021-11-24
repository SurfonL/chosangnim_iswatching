from utils.Workouts import Workouts

class BenchP(Workouts):
    def __init__(self):
        super().__init__()
        self._class_name = 'bench_down'
        self._pose_samples_folder = 'utils/pose_plots/bench'


    def run_bp(self,frame, landmarks):
        self.count(landmarks)
        #draw things
        #frame = draw_bp(frame)

        return frame, self.times


