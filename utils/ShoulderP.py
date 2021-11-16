import numpy as np
import cv2


pos = {'nose': 0, 'right_shoulder' : 11, 'right_elbow' : 13,'right_wrist' : 15,
            'left_shoulder' : 12,'left_elbow' : 14,'left_wrist' : 16, 'right_hip' : 23,'right_knee' : 25,
            'right_ankle' : 27, 'left_hip' : 24, 'left_knee' : 26, 'left_ankle' : 28 }
class ShoulderP:
    times = 0
    rate_r = 0
    rate_l = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    state = False

    @classmethod
    def sp_count(cls, f_coord, old_sp_state):
        # [x_coord,y_coord]
        wrist_r = f_coord[pos['right_wrist']]
        elbow_r = f_coord[pos['right_elbow']]
        shoulder_r = f_coord[pos['right_shoulder']]

        wrist_l = f_coord[pos['left_wrist']]
        elbow_l = f_coord[pos['left_elbow']]
        shoulder_l = f_coord[pos['left_shoulder']]

        #head_top = ('headtop',x_co,y_co)
        threshold = f_coord['nose'].y

        # boolean. define the position of each arm. True => up, False => down
        arm_r = cls.ud_state(wrist_r, elbow_r, threshold)
        arm_l = cls.ud_state(wrist_l, elbow_l, threshold)

        # if both arms are 'up' it is in up state.
        if arm_r and arm_l:
            ud = True
        else:
            ud = False

        #if [up] and [state before this was 'down'] then count.
        if ud and not old_sp_state:
            cls.times += 1
        else:
            pass

        propor_r = np.linalg.norm([wrist_r.x-elbow_r.x,wrist_r.y-elbow_r.y])
        propor_l = np.linalg.norm([wrist_l.x-elbow_l.x,wrist_l.y-elbow_l.y])

        # if in up-state, poi is rate towards down
        # if ud:
        #     cls.rate_r = (wrist_r[1] - threshold) / propor_r
        #     cls.rate_l = (wrist_l[1] - threshold) / propor_l
        #
        #
        # # if in down-state, poi is rate towards up
        # else:
        #     cls.rate_r = (threshold - elbow_r[1]) / propor_r
        #     cls.rate_l = (threshold - elbow_l[1]) / propor_l
        if propor_r > 0.1 and propor_l > 0.1:
            cls.rate_r = (threshold - elbow_r.y) / propor_r
            cls.rate_l = (threshold - elbow_l.y) / propor_l
        else:
            pass

        return ud

    @classmethod
    def draw_circle(cls, frame, landmark, frame_height, frame_width):
        right_wrist_x, right_wrist_y = landmark[pos['right_wrist']].x, landmark[pos['right_wrist']].y
        left_wrist_x, left_wrist_y = landmark[pos['left_wrist']].x, landmark[pos['left_wrist']].y
        right_wrist_x *= frame_width
        right_wrist_y *= frame_height
        left_wrist_x *= frame_width
        left_wrist_y *= frame_height

        lw, le, rw, re = cls.validity(landmark)

        if cls.rate_l > 0 and cls.rate_r > 0:
            if rw: frame = cls.image_alpha(frame, right_wrist_x, right_wrist_y, 30, (0, 255, 0), 0.3, 1, 1)
            if lw: frame = cls.image_alpha(frame, left_wrist_x, left_wrist_y, 30, (0, 255, 0), 0.3, 1, 1)
        else:
            if cls.rate_r > -1 and cls.rate_l > -1:
                if rw: frame = cls.image_alpha(frame, right_wrist_x, right_wrist_y, 30, (0, 255, 255), 0.3, 1-abs(cls.rate_r), 1)
                if lw: frame = cls.image_alpha(frame, left_wrist_x, left_wrist_y, 30, (0, 255, 255), 0.3, 1-abs(cls.rate_l), 1)
            else:
                if rw: frame = cls.image_alpha(frame, right_wrist_x, right_wrist_y, 30, (0, 255, 255), 0.3, 1, 1, fill = False)
                if lw: frame = cls.image_alpha(frame, left_wrist_x, left_wrist_y, 30, (0, 255, 255), 0.3, 1, 1, fill = False)
        return frame



    #팔 각도를 이용해 자세 교정
    #내려갈 때 속도 느리게 하기, 올라갈 때 빠르게 하기

    @staticmethod
    def sp_angle(wrist_coord, elbow_coord, shoulder_coord):
        to_wrist = wrist_coord - elbow_coord
        to_elbow = elbow_coord - shoulder_coord
        unit_vector_1 = to_wrist / np.linalg.norm(to_wrist)
        unit_vector_2 = to_elbow / np.linalg.norm(to_elbow)
        dot_product = np.dot(unit_vector_1, unit_vector_2)
        angle = np.arccos(dot_product)

        return angle

    @staticmethod
    def ud_state(wrist_coord, elbow_coord, threshold):
        elbow_y = elbow_coord.y
        wrist_y = wrist_coord.y
        # up state
        if elbow_y <= threshold:
            return True
        # down state
        elif wrist_y >= threshold:
            return False

    @staticmethod
    def image_alpha(frame, body_part_x, body_part_y, marker_radius, RGB, alpha, frame_count, fold_frame, fill = True):
        '''

            Args:
                body_part_x: coordinate_x (The center of circle)
                body_part_y: coordinate_y (The center of circle
                marker_radius: circle radius
                RGB: Tuple RGB color
                alpha: transparent intensity of circle

            Returns: Image with circle

            '''

        image_ = cv2.circle(frame, (int(body_part_x), int(body_part_y)), marker_radius, (255, 255, 255), 2)
        if fill:
            mask_rad = int(frame_count * marker_radius / (fold_frame))
            mask1 = np.ones_like(frame) * 255  # 흰 배경 검은 원
            mask1 = cv2.circle(mask1, (int(body_part_x), int(body_part_y)), mask_rad, (0, 0, 0), -1)
            mask2 = cv2.bitwise_not(mask1)  # 검은 배경 힌색 원
            mask_image = cv2.bitwise_and(image_, mask1)  # 원 외부의 이미지 가져오기
            blend1 = cv2.bitwise_and(image_, mask2)  # 원 내부의 이미지 가져오기
            blend2 = cv2.circle(mask2, (int(body_part_x), int(body_part_y)), mask_rad, RGB, -1)
            blended = mask_image + blend1 * (1 - alpha) + blend2 * alpha
            blended = blended.astype(np.uint8)
            return blended
        else:
            return image_
        #
        # blended = cv2.resize(cv2.flip(blended, 1), (1000, 1000))
        # return cv2.imshow('EfficientPose (Groos et al., 2020)', blended)

    @staticmethod
    def validity(landmark):
        lw= landmark[pos['left_wrist']].visibility >0.8
        le= landmark[pos['left_elbow']].visibility >0.8
        rw= landmark[pos['right_wrist']].visibility >0.8
        re= landmark[pos['right_elbow']].visibility >0.8
        return (lw,le,rw,re)


    @classmethod
    def run_shoulderp(cls, frame, landmarks):
        frame_height, frame_width = frame.shape[0], frame.shape[1]

        text = "Counts : " + str(cls.times)
        frame = cv2.putText(frame, text, (0, 100), cls.font, 1, (255, 255, 255), 2)
        frame = cls.draw_circle(frame, landmarks.landmark, frame_height, frame_width)



        return frame

    def draw_landmarks(
            image,
            landmarks,
            # upper_body_only,
            visibility_th=0.5,
    ):
        cv = cv2
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_point = []

        print(landmarks.landmark[0])

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