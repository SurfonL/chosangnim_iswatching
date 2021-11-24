from utils.Drawing import drawing
from utils.Workouts import Workouts

class Squat(Workouts):
    def __init__(self):
        self._class_name = 'squat'
        self.init('utils/pose_plots/squat')

    @classmethod
    def draw_circle(cls, frame, pose_predict, landmarks):
        frame_height, frame_width = frame.shape[0], frame.shape[1]
        right_hip = landmarks.landmark[23]
        left_hip = landmarks.landmark[24]
        if pose_predict > cls._enter_threshold:
            frame = drawing.image_alpha(frame, right_hip.x * frame_width, right_hip.y * frame_height, 30, (0, 255, 0), 0.3,
                                        1, 1)
            frame = drawing.image_alpha(frame, left_hip.x * frame_width, left_hip.y * frame_height, 30, (0, 255, 0), 0.3, 1,
                                        1)
        elif pose_predict > cls._exit_threshold:
            frame = drawing.image_alpha(frame, right_hip.x * frame_width, right_hip.y * frame_height, 30, (0, 255, 255),
                                        0.3, pose_predict - cls._exit_threshold, cls._enter_threshold - cls._exit_threshold)
            frame = drawing.image_alpha(frame, left_hip.x * frame_width, left_hip.y * frame_height, 30, (0, 255, 255),
                                        0.3, pose_predict - cls._exit_threshold, cls._enter_threshold - cls._exit_threshold)
        else:
            frame = drawing.image_alpha(frame, right_hip.x * frame_width, right_hip.y * frame_height, 30, (255, 255, 255), 0.3,
                                        1, 1, fill=False)
            frame = drawing.image_alpha(frame, left_hip.x * frame_width, left_hip.y * frame_height, 30, (255, 255, 255), 0.3,
                                        1, 1, fill=False)
        return frame

    def run_sq(self, frame, landmarks, landmarks_np):

        pose_knn = self.pose_classifier(landmarks_np)

        pose_predict = self.smoother(pose_knn)

        self.count(pose_predict)

        self.draw_circle(frame, pose_predict[self._class_name], landmarks)
        # draw things
        # frame = draw_bp(frame)

        return pose_predict

