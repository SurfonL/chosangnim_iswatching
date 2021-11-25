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
        mp_pose = mp.solutions.pose

        mp_drawing.draw_landmarks(
            image,
            landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())