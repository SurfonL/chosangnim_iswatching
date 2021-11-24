import numpy as np
import cv2
from utils.Workouts import Workouts
from utils.Drawing import drawing

pos = {'nose': 0, 'right_shoulder' : 11, 'right_elbow' : 13,'right_wrist' : 15,
            'left_shoulder' : 12,'left_elbow' : 14,'left_wrist' : 16, 'right_hip' : 23,'right_knee' : 25,
            'right_ankle' : 27, 'left_hip' : 24, 'left_knee' : 26, 'left_ankle' : 28 }
class ShoulderP(Workouts):
    def __init__(self):

        self.times = 0
        self.rate_r = 0
        self.rate_l = 0
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.state = False

        _pose_samples_folder = 'utils/pose_plots/shoulder'
        self.init('utils/pose_plots/shoulder')


    def sp_count(self, landmark, old_sp_state):
        # [x_coord,y_coord]
        wrist_r = landmark[pos['right_wrist']]
        elbow_r = landmark[pos['right_elbow']]
        shoulder_r = landmark[pos['right_shoulder']]

        wrist_l = landmark[pos['left_wrist']]
        elbow_l = landmark[pos['left_elbow']]
        shoulder_l = landmark[pos['left_shoulder']]

        #head_top = ('headtop',x_co,y_co)
        threshold = landmark[pos['nose']].y

        # boolean. define the position of each arm. True => up, False => down
        arm_r = self.ud_state(wrist_r, elbow_r, threshold)
        arm_l = self.ud_state(wrist_l, elbow_l, threshold)

        # if both arms are 'up' it is in up state.
        if arm_r and arm_l:
            ud = True
        else:
            ud = False

        #if [up] and [state before this was 'down'] then count.
        if ud and not old_sp_state:
            self.times += 1
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
            self.rate_r = (threshold - elbow_r.y) / propor_r
            self.rate_l = (threshold - elbow_l.y) / propor_l
        else:
            pass

        return ud

    def draw_circle(self, frame, landmark, frame_height, frame_width):
        right_wrist_x, right_wrist_y = landmark[pos['right_wrist']].x, landmark[pos['right_wrist']].y
        left_wrist_x, left_wrist_y = landmark[pos['left_wrist']].x, landmark[pos['left_wrist']].y
        right_wrist_x *= frame_width
        right_wrist_y *= frame_height
        left_wrist_x *= frame_width
        left_wrist_y *= frame_height

        lw, le, rw, re = self.validity(landmark)

        if self.rate_l > 0 and self.rate_r > 0:
            if rw: frame = drawing.image_alpha(frame, right_wrist_x, right_wrist_y, 30, (0, 255, 0), 0.3, 1, 1)
            if lw: frame = drawing.image_alpha(frame, left_wrist_x, left_wrist_y, 30, (0, 255, 0), 0.3, 1, 1)
        else:
            if self.rate_r > -1 and self.rate_l > -1:
                if rw: frame = drawing.image_alpha(frame, right_wrist_x, right_wrist_y, 30, (0, 255, 255), 0.3, 1-abs(self.rate_r), 1)
                if lw: frame = drawing.image_alpha(frame, left_wrist_x, left_wrist_y, 30, (0, 255, 255), 0.3, 1-abs(self.rate_l), 1)
            else:
                if rw: frame = drawing.image_alpha(frame, right_wrist_x, right_wrist_y, 30, (0, 255, 255), 0.3, 1, 1, fill = False)
                if lw: frame = drawing.image_alpha(frame, left_wrist_x, left_wrist_y, 30, (0, 255, 255), 0.3, 1, 1, fill = False)
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
        lw= landmark[pos['left_wrist']].visibility >0
        le= landmark[pos['left_elbow']].visibility >0
        rw= landmark[pos['right_wrist']].visibility >0
        re= landmark[pos['right_elbow']].visibility >0
        return (lw,le,rw,re)


    def run_sp(self, frame, landmarks, landmarks_np):
        frame_height, frame_width = frame.shape[0], frame.shape[1]

        val = self.validity(landmarks.landmark)
        if all(val):
            self.state = self.sp_count(landmarks.landmark, self.state)
        else:
            pass

        frame = self.draw_circle(frame, landmarks.landmark, frame_height, frame_width)

        pose_knn = self.pose_classifier(landmarks_np)
        pose_predict = self.smoother(pose_knn)

        return pose_predict

