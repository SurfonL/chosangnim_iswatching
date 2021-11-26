import cv2
import numpy as np
import mediapipe as mp

class drawing:



    @staticmethod
    def image_alpha(frame, body_part_x, body_part_y, marker_radius, RGB, alpha, frame_count, fold_frame, fill=True):
        '''

            Args:
                body_part_x: coordinate_x (The center of circle)
                body_part_y: coordinate_y (The center of circle
                marker_radius: circle radius
                RGB: Tuple RGB color
                alpha: transparent intensity of circle
                frame_count: present state(probability)
                fold_frame: threshold of the color change
                fill: default is filling inside the circle. If you want to draw only circle line, set 'False'

            Returns: Image with circle

            '''

        image = cv2.circle(frame, (int(body_part_x), int(body_part_y)), marker_radius, (255, 255, 255), 2)
        if fill:
            mask_rad = int(frame_count * marker_radius / (fold_frame))
            mask1 = np.ones_like(frame) * 255  # 흰 배경 검은 원
            print(mask_rad)
            mask1 = cv2.circle(mask1, (int(body_part_x), int(body_part_y)), mask_rad, (0, 0, 0), -1)
            mask2 = cv2.bitwise_not(mask1)  # 검은 배경 힌색 원
            mask_image = cv2.bitwise_and(image, mask1)  # 원 외부의 이미지 가져오기
            blend1 = cv2.bitwise_and(image, mask2)  # 원 내부의 이미지 가져오기
            blend2 = cv2.circle(mask2, (int(body_part_x), int(body_part_y)), mask_rad, RGB, -1)
            image = mask_image + blend1 * (1 - alpha) + blend2 * alpha
            image = image.astype(np.uint8)

        return image

    @staticmethod
    def annotation(image, landmarks):
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        full_connection = [(15, 21), (16, 20), (18, 20), (3, 7), (14, 16), (23, 25), (28, 30), (11, 23), (27, 31), (6, 8), (15, 17), (24, 26), (16, 22), (4, 5), (5, 6), (29, 31), (12, 24), (23, 24), (
0, 1), (9, 10), (1, 2), (0, 4), (11, 13), (30, 32), (28, 32), (15, 19), (16, 18), (25, 27), (26, 28), (12, 14), (17, 19), (2, 3), (11, 12), (27, 29), (13, 15)]

        mp_drawing.draw_landmarks(
            image,
            landmarks,
            full_connection,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    @staticmethod
    def draw_lines(image, landmarks, add_list):
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        full_connection = [(15, 21), (16, 20), (18, 20), (3, 7), (14, 16), (23, 25), (28, 30), (11, 23), (27, 31),
                           (6, 8), (15, 17), (24, 26), (16, 22), (4, 5), (5, 6), (29, 31), (12, 24), (23, 24), (
                               0, 1), (9, 10), (1, 2), (0, 4), (11, 13), (30, 32), (28, 32), (15, 19), (16, 18),
                           (25, 27), (26, 28), (12, 14), (17, 19), (2, 3), (11, 12), (27, 29), (13, 15)]
        # [landmarks.landmark[i] for i in range(len(landmarks.landmark)) if i not in remove_list]

        connection = []
        for i in range(len(full_connection)):
            a,b = full_connection[i]
            if (a in add_list) or (b in add_list):
                connection.append(full_connection[i])

        mp_drawing.draw_landmarks(
            image,
            landmarks,
            connection,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    @staticmethod
    def draw_landmarks_making(
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
            landmark_point.append([landmark.visibility, (landmark_x, landmark_y)])

            if landmark.visibility < visibility_th:
                continue

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



    @staticmethod
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
