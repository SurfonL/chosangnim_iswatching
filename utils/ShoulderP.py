import numpy as np
import cv2
from utils.KnnClassif import FullBodyPoseEmbedder, PoseClassifier, EMADictSmoothing
from utils.Drawing import drawing

pos = {'nose': 0, 'right_shoulder' : 11, 'right_elbow' : 13,'right_wrist' : 15,
            'left_shoulder' : 12,'left_elbow' : 14,'left_wrist' : 16, 'right_hip' : 23,'right_knee' : 25,
            'right_ankle' : 27, 'left_hip' : 24, 'left_knee' : 26, 'left_ankle' : 28,
       'left_index': 19, 'right_index': 20}
class ShoulderP:
    _exit_threshold = 4

    times = 0
    rate_r = 0
    rate_l = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    state = False

    pose_embedder = FullBodyPoseEmbedder()
    pose_classifier = PoseClassifier(
        pose_samples_folder='utils/pose_plots/shoulder',
        pose_embedder=pose_embedder,
        top_n_by_max_distance=30,
        top_n_by_mean_distance=10)
    smoother = EMADictSmoothing('utils/pose_plots/shoulder')


    @classmethod
    def sp_count(cls, landmark, old_sp_state):
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
        right_index_x, right_index_y = landmark[pos['right_index']].x, landmark[pos['right_index']].y
        left_index_x, left_index_y = landmark[pos['left_index']].x, landmark[pos['left_index']].y
        right_index_x *= frame_width
        right_index_y *= frame_height
        left_index_x *= frame_width
        left_index_y *= frame_height

        lw, le, rw, re = cls.validity(landmark)

        if cls.rate_l > 0 and cls.rate_r > 0:
            if rw: frame = drawing.image_alpha(frame, right_index_x, right_index_y, 30, (0, 255, 0), 0.3, 1, 1)
            if lw: frame = drawing.image_alpha(frame, left_index_x, left_index_y, 30, (0, 255, 0), 0.3, 1, 1)
        else:
            if cls.rate_r > -1 and cls.rate_l > -1:
                if rw: frame = drawing.image_alpha(frame, right_index_x, right_index_y, 30, (0, 255, 255), 0.3, 1-abs(cls.rate_r), 1)
                if lw: frame = drawing.image_alpha(frame, left_index_x, left_index_y, 30, (0, 255, 255), 0.3, 1-abs(cls.rate_l), 1)
            else:
                if rw: frame = drawing.image_alpha(frame, right_index_x, right_index_y, 30, (0, 255, 255), 0.3, 1, 1, fill = False)
                if lw: frame = drawing.image_alpha(frame, left_index_x, left_index_y, 30, (0, 255, 255), 0.3, 1, 1, fill = False)
        return frame



    #??? ????????? ????????? ?????? ??????
    #????????? ??? ?????? ????????? ??????, ????????? ??? ????????? ??????

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
            mask1 = np.ones_like(frame) * 255  # ??? ?????? ?????? ???
            mask1 = cv2.circle(mask1, (int(body_part_x), int(body_part_y)), mask_rad, (0, 0, 0), -1)
            mask2 = cv2.bitwise_not(mask1)  # ?????? ?????? ?????? ???
            mask_image = cv2.bitwise_and(image_, mask1)  # ??? ????????? ????????? ????????????
            blend1 = cv2.bitwise_and(image_, mask2)  # ??? ????????? ????????? ????????????
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
        lw= landmark[pos['left_wrist']].visibility >0.3
        le= landmark[pos['left_elbow']].visibility >0.3
        rw= landmark[pos['right_wrist']].visibility >0.3
        re= landmark[pos['right_elbow']].visibility >0.3
        return (lw,le,rw,re)

    @classmethod
    def set_param(cls, enter, exit, win ,a):
        cls._enter_threshold = enter
        cls._exit_threshold = exit
        cls.smoother.set_rate(win, a)

    @classmethod
    def run_sp(cls, frame, pose_predict, landmarks, locked = False):
        frame_height, frame_width = frame.shape[0], frame.shape[1]
        if locked:
            frame_height, frame_width = frame.shape[0], frame.shape[1]
            landmarks_np = np.array([[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                                     for lmk in landmarks.landmark], dtype=np.float32)
            pose_classification = cls.pose_classifier(landmarks_np)
            pose_predict = cls.smoother(pose_classification)

            frame = cls.draw_circle(frame, landmarks.landmark, frame_height, frame_width)

        # else:
        #     frame = drawing.annotation(frame, landmarks)

        if all(cls.validity(landmarks.landmark)):
            cls.state = cls.sp_count(landmarks.landmark, cls.state)
        else:
            pass

        return frame, pose_predict
