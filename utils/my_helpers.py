import pandas as pd
import time

from utils.KnnClassif import FullBodyPoseEmbedder, PoseClassifier, EMADictSmoothing
import mediapipe as mp
import cv2
import numpy as np

class StandardProcess:

    def __init__(self, model_complexity):
        self.pose = mp.solutions.pose.Pose(model_complexity=model_complexity)
        self.pose_embedder = FullBodyPoseEmbedder()
        self.pose_classifier = PoseClassifier(
            pose_samples_folder='utils/fitness_poses_csvs_out',
            pose_embedder=self.pose_embedder,
            top_n_by_max_distance=30,
            top_n_by_mean_distance=10)


    def std_process(self, frame, width = None, height = None):
        # (480 640 3)
        frame = frame.to_ndarray(width= width, height = height, format="bgr24")
        # batch인데 어차피 1임
        frame = cv2.flip(frame,1)
        results = self.pose.process(frame)
        landmarks = results.pose_landmarks

        self.frame_height, self.frame_width, _ = frame.shape

        return frame, landmarks, self.frame_height, self.frame_width

    def pose_class(self, landmarks, n_min, n_max):
        landmarks_np = np.array([[lmk.x * self.frame_width, lmk.y * self.frame_height, lmk.z * self.frame_width]
                                 for lmk in landmarks.landmark], dtype=np.float32)
        self.pose_classifier.set_minmaxn(n_min, n_max)
        pose_classification = self.pose_classifier(landmarks_np)

        return pose_classification

def print_count(frame,height,width,count, goal, pose, pose_prob, w_time, r_time,rest_thresh, debug=True):
    font_color = (255,255,255)
    if pose == 'bench':
        pose = 'bench press'

    if goal != 0:
        count = goal-count
        text = str(count) + " to go"
    else:
        text = "Count: " + str(count)
    f_size = height / 200
    f_thick = int(f_size * 1.5)
    t_size, t_y = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, f_size, f_thick)
    frame = cv2.putText(frame, text, (width - t_size[0] - int(width / 20), t_size[1] + int(height / 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, f_size, font_color, f_thick)

    f_size = height/400
    f_thick = f_thick -1
    t_size, t_y2 = cv2.getTextSize(pose, cv2.FONT_HERSHEY_SIMPLEX, f_size, f_thick)
    frame = cv2.putText(frame, pose, (width - t_size[0] - int(width / 20), t_size[1] + int(height / 15) +t_y*3),
                        cv2.FONT_HERSHEY_SIMPLEX, f_size, font_color, f_thick)

    now = time.time()
    if pose == 'resting':
        r_time = now if r_time < rest_thresh else r_time+rest_thresh
        t = str(round(now-r_time,1))
    else:
        t = str(round(now-w_time,1))
    f_size = height/400
    f_thick = f_thick -1
    t_size, _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, f_size, f_thick)
    frame = cv2.putText(frame, t, (width - t_size[0] - int(width / 20), t_size[1] + int(height / 15) +t_y*3+t_y2*4),
                        cv2.FONT_HERSHEY_SIMPLEX, f_size, font_color, f_thick)

    if debug:
        f_size = height / 600
        f_thick = f_thick - 1 if f_thick >1 else 1
        t_size, _ = cv2.getTextSize(pose_prob, cv2.FONT_HERSHEY_SIMPLEX, f_size, f_thick)
        frame = cv2.putText(frame, pose_prob,
                            (width - t_size[0] - int(width / 20), t_size[1] + int(height / 15) + t_y * 3 + t_y2*7),
                            cv2.FONT_HERSHEY_SIMPLEX, f_size, font_color, f_thick)

    return frame

def workout_row(set_no, pose, count, set_duration, rest_duration):
    columns = ['pose', 'count','set duration', 'rest duration']
    row = {'Set No: {}'.format(set_no): [pose, count, set_duration, rest_duration]}

    return pd.DataFrame.from_dict(row, orient='index', columns = columns)

def draw_landmarks(
        image,
        landmarks,
        # upper_body_only,
        visibility_th=0.5,
):
    cv = cv2
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []


    for index, landmark in enumerate(landmarks.landmark):

        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z
        landmark_point.append([landmark.visibility, (landmark_x, landmark_y)])

        if landmark.visibility < visibility_th:
            continue

        if index == 0:  # 鼻
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 1:  # 右目：目頭
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 2:  # 右目：瞳
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 3:  # 右目：目尻
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 4:  # 左目：目頭
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 5:  # 左目：瞳
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 6:  # 左目：目尻
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 7:  # 右耳
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 8:  # 左耳
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 9:  # 口：左端
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 10:  # 口：左端
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 11:  # 右肩
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 12:  # 左肩
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 13:  # 右肘
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 14:  # 左肘
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 15:  # 右手首
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 16:  # 左手首
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 17:  # 右手1(外側端)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 18:  # 左手1(外側端)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 19:  # 右手2(先端)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 20:  # 左手2(先端)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 21:  # 右手3(内側端)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 22:  # 左手3(内側端)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 23:  # 腰(右側)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 24:  # 腰(左側)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 25:  # 右ひざ
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 26:  # 左ひざ
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 27:  # 右足首
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 28:  # 左足首
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 29:  # 右かかと
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 30:  # 左かかと
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 31:  # 右つま先
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 32:  # 左つま先
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)

        # if not upper_body_only:
        if True:
            cv.putText(image, "z:" + str(round(landmark_z, 3)),
                       (landmark_x - 10, landmark_y - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                       cv.LINE_AA)

    # 右目
    if landmark_point[1][0] > visibility_th and landmark_point[2][
        0] > visibility_th:
        cv.line(image, landmark_point[1][1], landmark_point[2][1],
                (0, 255, 0), 2)
    if landmark_point[2][0] > visibility_th and landmark_point[3][
        0] > visibility_th:
        cv.line(image, landmark_point[2][1], landmark_point[3][1],
                (0, 255, 0), 2)

    # 左目
    if landmark_point[4][0] > visibility_th and landmark_point[5][
        0] > visibility_th:
        cv.line(image, landmark_point[4][1], landmark_point[5][1],
                (0, 255, 0), 2)
    if landmark_point[5][0] > visibility_th and landmark_point[6][
        0] > visibility_th:
        cv.line(image, landmark_point[5][1], landmark_point[6][1],
                (0, 255, 0), 2)

    # 口
    if landmark_point[9][0] > visibility_th and landmark_point[10][
        0] > visibility_th:
        cv.line(image, landmark_point[9][1], landmark_point[10][1],
                (0, 255, 0), 2)

    # 肩
    if landmark_point[11][0] > visibility_th and landmark_point[12][
        0] > visibility_th:
        cv.line(image, landmark_point[11][1], landmark_point[12][1],
                (0, 255, 0), 2)

    # 右腕
    if landmark_point[11][0] > visibility_th and landmark_point[13][
        0] > visibility_th:
        cv.line(image, landmark_point[11][1], landmark_point[13][1],
                (0, 255, 0), 2)
    if landmark_point[13][0] > visibility_th and landmark_point[15][
        0] > visibility_th:
        cv.line(image, landmark_point[13][1], landmark_point[15][1],
                (0, 255, 0), 2)

    # 左腕
    if landmark_point[12][0] > visibility_th and landmark_point[14][
        0] > visibility_th:
        cv.line(image, landmark_point[12][1], landmark_point[14][1],
                (0, 255, 0), 2)
    if landmark_point[14][0] > visibility_th and landmark_point[16][
        0] > visibility_th:
        cv.line(image, landmark_point[14][1], landmark_point[16][1],
                (0, 255, 0), 2)

    # 右手
    if landmark_point[15][0] > visibility_th and landmark_point[17][
        0] > visibility_th:
        cv.line(image, landmark_point[15][1], landmark_point[17][1],
                (0, 255, 0), 2)
    if landmark_point[17][0] > visibility_th and landmark_point[19][
        0] > visibility_th:
        cv.line(image, landmark_point[17][1], landmark_point[19][1],
                (0, 255, 0), 2)
    if landmark_point[19][0] > visibility_th and landmark_point[21][
        0] > visibility_th:
        cv.line(image, landmark_point[19][1], landmark_point[21][1],
                (0, 255, 0), 2)
    if landmark_point[21][0] > visibility_th and landmark_point[15][
        0] > visibility_th:
        cv.line(image, landmark_point[21][1], landmark_point[15][1],
                (0, 255, 0), 2)

    # 左手
    if landmark_point[16][0] > visibility_th and landmark_point[18][
        0] > visibility_th:
        cv.line(image, landmark_point[16][1], landmark_point[18][1],
                (0, 255, 0), 2)
    if landmark_point[18][0] > visibility_th and landmark_point[20][
        0] > visibility_th:
        cv.line(image, landmark_point[18][1], landmark_point[20][1],
                (0, 255, 0), 2)
    if landmark_point[20][0] > visibility_th and landmark_point[22][
        0] > visibility_th:
        cv.line(image, landmark_point[20][1], landmark_point[22][1],
                (0, 255, 0), 2)
    if landmark_point[22][0] > visibility_th and landmark_point[16][
        0] > visibility_th:
        cv.line(image, landmark_point[22][1], landmark_point[16][1],
                (0, 255, 0), 2)

    # 胴体
    if landmark_point[11][0] > visibility_th and landmark_point[23][
        0] > visibility_th:
        cv.line(image, landmark_point[11][1], landmark_point[23][1],
                (0, 255, 0), 2)
    if landmark_point[12][0] > visibility_th and landmark_point[24][
        0] > visibility_th:
        cv.line(image, landmark_point[12][1], landmark_point[24][1],
                (0, 255, 0), 2)
    if landmark_point[23][0] > visibility_th and landmark_point[24][
        0] > visibility_th:
        cv.line(image, landmark_point[23][1], landmark_point[24][1],
                (0, 255, 0), 2)

    if len(landmark_point) > 25:
        # 右足
        if landmark_point[23][0] > visibility_th and landmark_point[25][
            0] > visibility_th:
            cv.line(image, landmark_point[23][1], landmark_point[25][1],
                    (0, 255, 0), 2)
        if landmark_point[25][0] > visibility_th and landmark_point[27][
            0] > visibility_th:
            cv.line(image, landmark_point[25][1], landmark_point[27][1],
                    (0, 255, 0), 2)
        if landmark_point[27][0] > visibility_th and landmark_point[29][
            0] > visibility_th:
            cv.line(image, landmark_point[27][1], landmark_point[29][1],
                    (0, 255, 0), 2)
        if landmark_point[29][0] > visibility_th and landmark_point[31][
            0] > visibility_th:
            cv.line(image, landmark_point[29][1], landmark_point[31][1],
                    (0, 255, 0), 2)

        # 左足
        if landmark_point[24][0] > visibility_th and landmark_point[26][
            0] > visibility_th:
            cv.line(image, landmark_point[24][1], landmark_point[26][1],
                    (0, 255, 0), 2)
        if landmark_point[26][0] > visibility_th and landmark_point[28][
            0] > visibility_th:
            cv.line(image, landmark_point[26][1], landmark_point[28][1],
                    (0, 255, 0), 2)
        if landmark_point[28][0] > visibility_th and landmark_point[30][
            0] > visibility_th:
            cv.line(image, landmark_point[28][1], landmark_point[30][1],
                    (0, 255, 0), 2)
        if landmark_point[30][0] > visibility_th and landmark_point[32][
            0] > visibility_th:
            cv.line(image, landmark_point[30][1], landmark_point[32][1],
                    (0, 255, 0), 2)
    return image



