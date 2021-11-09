import numpy as np
import cv2


pos = {'head_top': 0, 'upper_neck': 1, 'right_shoulder' : 2, 'right_elbow' : 3,'right_wrist' : 4, 'thorax' : 5,
            'left_shoulder' : 6,'left_elbow' : 7,'left_wrist' : 8,'pelvis' : 9,'right_hip' : 10,'right_knee' : 11,
            'right_ankle' : 12, 'left_hip' : 13, 'left_knee' : 14, 'left_ankle' : 15 }
class ShoulderP:
    times = 0
    rate_r = 0
    rate_l = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    state = False

    @classmethod
    def sp_count(cls, f_coord, old_sp_state):
        # [x_coord,y_coord]
        wrist_r = f_coord[pos['right_wrist']][1:]
        elbow_r = f_coord[pos['right_elbow']][1:]
        shoulder_r = f_coord[pos['right_shoulder']][1:]

        wrist_l = f_coord[pos['left_wrist']][1:]
        elbow_l = f_coord[pos['left_elbow']][1:]
        shoulder_l = f_coord[pos['left_shoulder']][1:]

        #head_top = ('headtop',x_co,y_co)
        threshold = (f_coord[pos['head_top']][2] + f_coord[pos['upper_neck']][2]) / 2

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

        propor_r = np.linalg.norm([wrist_r[0]-elbow_r[0],wrist_r[1]-elbow_r[1]])
        propor_l = np.linalg.norm([wrist_l[0]-elbow_l[0],wrist_l[1]-elbow_l[1]])

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
            cls.rate_r = (threshold - elbow_r[1]) / propor_r
            cls.rate_l = (threshold - elbow_l[1]) / propor_l
        else:
            pass

        return ud

    @classmethod
    def draw_circle(cls, frame, f_coord, frame_width, frame_height):
        _, right_wrist_x, right_wrist_y = f_coord[pos['right_wrist']]
        _, left_wrist_x, left_wrist_y = f_coord[pos['left_wrist']]
        right_wrist_x *= frame_width
        right_wrist_y *= frame_height
        left_wrist_x *= frame_width
        left_wrist_y *= frame_height

        lw, le, rw, re = cls.validity(f_coord)

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
        elbow_y = elbow_coord[1]
        wrist_y = wrist_coord[1]
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
    def validity(f_coord):
        lw= f_coord[pos['left_wrist']][1] >0
        le= f_coord[pos['left_elbow']][1] >0
        rw= f_coord[pos['right_wrist']][1] >0
        re= f_coord[pos['right_elbow']][1] >0
        return (lw,le,rw,re)


    @classmethod
    def run_shoulderp(cls, frame, f_coord, frame_width, frame_height):
        val = cls.validity(f_coord)
        if all(val):
            cls.state = cls.sp_count(f_coord, cls.state)
        else:
            # print('not valid')
            pass
        text = "Counts : " + str(cls.times)
        frame = cv2.putText(frame, text, (0, 100), cls.font, 1, (255, 255, 255), 2)
        frame = cls.draw_circle(frame, f_coord, frame_width, frame_height)
        return frame
