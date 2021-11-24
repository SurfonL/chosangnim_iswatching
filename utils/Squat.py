from utils.Drawing import drawing

class Squat:
    _class_name = 'squat_down'

    # If pose counter passes given threshold, then we enter the pose.
    _enter_threshold = 6
    _exit_threshold = 4

    # Either we are in given pose or not.
    _pose_entered = False

    # Number of times we exited the pose.
    times = 0

    @classmethod
    def count_sq(cls, pose_classification):
        """Counts number of repetitions happend until given frame.

        We use two thresholds. First you need to go above the higher one to enter
        the pose, and then you need to go below the lower one to exit it. Difference
        between the thresholds makes it stable to prediction jittering (which will
        cause wrong counts in case of having only one threshold).

        Args:
          pose_classification: Pose classification dictionary on current frame.
            Sample:
              {
                'pushups_down': 8.3,
                'pushups_up': 1.7,
              }

        Returns:
          Integer counter of repetitions.
        """
        # Get pose confidence.
        pose_confidence = 0.0
        if cls._class_name in pose_classification:
            pose_confidence = pose_classification[cls._class_name]

        # On the very first frame or if we were out of the pose, just check if we
        # entered it on this frame and update the state.
        if not cls._pose_entered:
            cls._pose_entered = pose_confidence > cls._enter_threshold
            return cls.times

        # If we were in the pose and are exiting it, then increase the counter and
        # update the state.
        if pose_confidence < cls._exit_threshold:
            cls.times += 1
            cls._pose_entered = False

    @classmethod
    def set_thresh(cls,enter,exit):
        cls._enter_threshold=enter
        cls._exit_threshold=exit

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
                                        0.3, pose_predict - cls._exit_threshold, cls._enter_threshold - cls._enter_threshold)
            frame = drawing.image_alpha(frame, left_hip.x * frame_width, left_hip.y * frame_height, 30, (0, 255, 255),
                                        0.3, pose_predict - cls._exit_threshold, cls._enter_threshold - cls._enter_threshold)
        else:
            frame = drawing.image_alpha(frame, right_hip.x * frame_width, right_hip.y * frame_height, 30, (255, 255, 255), 0.3,
                                        1, 1, fill=False)
            frame = drawing.image_alpha(frame, left_hip.x * frame_width, left_hip.y * frame_height, 30, (255, 255, 255), 0.3,
                                        1, 1, fill=False)
        return frame

    @classmethod
    def run_sq(cls, frame, pose_predict, landmarks):
        cls.count_sq(pose_predict)
        cls.draw_circle(frame, pose_predict[cls._class_name], landmarks)
        # draw things
        # frame = draw_bp(frame)

        return frame, cls.times

