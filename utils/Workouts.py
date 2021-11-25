from utils.KnnClassif import FullBodyPoseEmbedder, PoseClassifier, EMADictSmoothing

class Workouts:
    def __init__(self):
        self._class_name = None
        self._pose_samples_folder= None

        # If pose counter passes given threshold, then we enter the pose.
        self._enter_threshold = 6
        self._exit_threshold = 4

        # Either we are in given pose or not.
        self._pose_entered = False

        # Number of times we exited the pose.
        self.times = 0


        self.pose_embedder = FullBodyPoseEmbedder()
        self.pose_classifier = PoseClassifier(
            pose_samples_folder=self._pose_samples_folder,
            pose_embedder=self.pose_embedder,
            top_n_by_max_distance=30,
            top_n_by_mean_distance=10)


    def count(self, pose_classification):
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
        if self._class_name in pose_classification:
            pose_confidence = pose_classification[self._class_name]

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

    @classmethod
    def set_thresh(cls,enter,exit):
        self._enter_threshold=enter
        self._exit_threshold=exit