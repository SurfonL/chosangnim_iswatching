from utils.Workouts import Workouts

class BenchP(Workouts):
    def __init__(self):
        super().__init__()
        self._class_name = 'bench_down'
        self._pose_samples_folder = 'utils/pose_plots/bench'

    @classmethod
    def run_bp(cls,frame,pose_predict, landmarks):


        cls.count(pose_predict)
        #draw things
        #frame = draw_bp(frame)

        return frame, cls.times


