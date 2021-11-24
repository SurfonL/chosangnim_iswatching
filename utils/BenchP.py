from utils.Workouts import Workouts
from utils.Drawing import drawing

class BenchP(Workouts):
    def __init__(self):
        self._enter_threshold = 6
        self._exit_threshold = 4
        self._class_name = 'bench'
        self.init('utils/pose_plots/bench')
        self.times =0
        
        
    def count(self, pose_predict):
       
        # Get pose confidence.
        pose_confidence = 0.0
        if self._class_name in pose_predict:
            pose_confidence = pose_predict[self._class_name]

        # On the very first frame or if we were out of the pose, just check if we
        # entered it on this frame and update the state.
        if not self._pose_entered:
            self._pose_entered = pose_confidence > self._enter_threshold
            return self.times

        # If we were in the pose and are exiting it, then increase the counter and
        # update the state.
        if pose_confidence < self._exit_threshold:
            self.times += 1
            self._pose_entered = False


    def run_bp(self,frame, landmarks, landmarks_np):
        pose_knn = self.pose_classifier(landmarks_np)
        pose_predict = self.smoother(pose_knn)
        self.count(pose_predict)
        #draw things
        #frame = draw_bp(frame)

        return pose_predict


