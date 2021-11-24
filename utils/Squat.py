from utils.Drawing import drawing
from utils.Workouts import Workouts

class Squat(Workouts):
    def __init__(self):
        super().__init__()
        self._class_name = 'squat_down'
        self._pose_samples_folder = 'utils/pose_plots/squat'

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

    def run_sq(self, frame, pose_predict, landmarks):
        self.count(pose_predict)
        self.draw_circle(frame, landmarks)
        # draw things
        # frame = draw_bp(frame)

        return frame, self.times

