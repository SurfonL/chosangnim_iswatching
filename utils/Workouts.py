from utils.KnnClassif import FullBodyPoseEmbedder, PoseClassifier, EMADictSmoothing

class Workouts:

    _class_name = None
    _pose_samples_folder= None


    # If pose counter passes given threshold, then we enter the pose.
    _enter_threshold = 6
    _exit_threshold = 4

    _window = 10
    _alpha = 0.1

    _max_d = 30
    _min_d = 10


    # Either we are in given pose or not.
    _pose_entered = False

    # Number of times we exited the pose.
    times = 0

    csv_dir = None


    pose_embedder = FullBodyPoseEmbedder()


    @classmethod
    def init(cls, _pose_samples_folder):
        cls.smoother = EMADictSmoothing(_pose_samples_folder,
                                window_size=cls._window,
                                     alpha=cls._alpha)
        cls.pose_classifier = PoseClassifier(
            pose_samples_folder=_pose_samples_folder,
            pose_embedder=cls.pose_embedder,
            top_n_by_max_distance=cls._max_d,
            top_n_by_mean_distance=cls._min_d)


    @classmethod
    def count(cls, pose_predict):

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
        if cls._class_name in pose_predict:
            pose_confidence = pose_predict[cls._class_name]

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
    def set_smooth(cls, win, alpha):
        cls. _window = win
        cls._alpha = alpha

    @classmethod
    def set_max_d(cls, win, alpha):
        cls._window = win
        cls._alpha = alpha