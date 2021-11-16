import cv2


def std_process(frame, pose):
    # (480 640 3)
    frame = frame.to_ndarray(width= 720, height = 480, format="bgr24")
    # batch인데 어차피 1임
    frame = cv2.flip(frame,1)
    results = pose.process(frame)
    landmarks = results.pose_landmarks

    return frame, landmarks