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
    def draw_lines(image, landmarks, remove_list):
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
            if (a not in remove_list) and (b not in remove_list):
                connection.append(full_connection[i])

        ds = mp_drawing_styles.get_default_pose_landmarks_style()
        for i in remove_list:
            ds.pop('<PoseLandmark.LEFT_KNEE: {}>'.format(i))

        mp_drawing.draw_landmarks(
            image,
            landmarks,
            connection,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    @staticmethod
    def get_landmark_keys():
        keys = ['<PoseLandmark.NOSE: 0>','<PoseLandmark.LEFT_EYE_INNER: 1>', '<PoseLandmark.LEFT_EYE: 2 >',
                '<PoseLandmark.LEFT_EYE_OUTER: 3>','<PoseLandmark.RIGHT_EYE_INNER: 4>','<PoseLandmark.RIGHT_EYE: 5>','<PoseLandmark.RIGHT_EYE_OUTER: 6>', '<PoseLandmark.LEFT_EAR: 7>' '<PoseLandmark.RIGHT_EAR: 8>',
        , '<PoseLandmark.MOUTH_LEFT: 9>', '<PoseLandmark.MOUTH_RIGHT: 10>', '<PoseLandmark.LEFT_SHOULDER: 11>',  '< PoseLandmark.RIGHT_SHOULDER:12>',
        '<PoseLandmark.LEFT_ELBOW: 13>',  '<PoseLandmark.RIGHT_ELBOW: 14>', '<PoseLandmark.LEFT_WRIST: 15>', '<PoseLandmark.RIGHT_WRIST: 16>', '<PoseLandmark.LEFT_PINKY: 17>',
        '<PoseLandmark.RIGHT_PINKY: 18>','<PoseLandmark.LEFT_INDEX: 19>', '<PoseLandmark.RIGHT_INDEX: 20>', '<PoseLandmark.LEFT_THUMB: 21>', '<PoseLandmark.RIGHT_THUMB: 22>', '<PoseLandmark.LEFT_HIP: 23>',
        '<PoseLandmark.LEFT_KNEE: 25>', '<PoseLandmark.LEFT_ANKLE: 27>', '<PoseLandmark.LEFT_HEEL: 29>',
        '<PoseLandmark.LEFT_FOOT_INDEX: 31>', '<PoseLandmark.RIGHT_FOOT_INDEX: 32>',


         '<PoseLandmark.RIGHT_HIP: 24>','<PoseLandmark.RIGHT_KNEE: 26>',
        '<PoseLandmark.RIGHT_ANKLE: 28>', '<PoseLandmark.RIGHT_HEEL: 30>', ]