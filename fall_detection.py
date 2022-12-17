import cv2
import time
import math as m
import mediapipe as mp


def findDistance(x1,y1,x2,y2):
    return m.sqrt((x2-x1)**2+(y2-y1)**2)
def findAngle(x1, y1, x2, y2):
    if x2 == x1 and y2 == y1:
        return 90
    theta = m.acos(-(y2-y1)/(0.001+m.sqrt((x2-x1)**2+(y2-y1)**2)))
    return int(180/m.pi)*theta
def sendWarning(x):
    pass

good_frames = 0
bad_frames = 0

font = cv2.FONT_HERSHEY_SIMPLEX

blue = (255, 127, 0)
red = (50, 50, 255)
green = (127, 255, 0)
dark_blue = (127, 20, 0)
light_green = (127, 233, 100)
yellow = (0, 255, 255)
pink = (255, 0, 255)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mpDraw = mp.solutions.drawing_utils

filename = 'queda.mp4'
cap = cv2.VideoCapture(filename)

while True:
    success, img = cap.read()
    img = cv2.resize(img, (int(img.shape[1]*1.2), int(img.shape[0]*1.2)))
    if not success:
        print("NULL frame")
        break

    fps = cap.get(cv2.CAP_PROP_FPS)
    h, w = img.shape[:2]
    # cv2.line(img, (w//2, 0), (w//2, h), red, 2)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    keypoints = pose.process(imgRGB)

    lm = keypoints.pose_landmarks
    # print(lm)
    lmPose = mp_pose.PoseLandmark
    if lm:
        # print(lm.landmark)
        # left shoulder
        l_sh_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
        l_sh_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)

        # right shoulder
        r_sh_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
        r_sh_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)

        avg_sh_x = int(l_sh_x + r_sh_x)/2
        avg_sh_y = int(l_sh_y + r_sh_y)/2
        # left hip
        l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
        l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)

        # right_hip
        r_hip_x = int(lm.landmark[lmPose.RIGHT_HIP].x * w)
        r_hip_y = int(lm.landmark[lmPose.RIGHT_HIP].y * h)

        avg_hip_x = int(l_hip_x + r_hip_x) / 2
        avg_hip_y = int(l_hip_y + r_hip_y) / 2

        # print(f'({avg_sh_x}, {avg_sh_y}), ({avg_hip_x}, {avg_hip_y})')

        vector_body = (-avg_sh_x+avg_hip_x, -avg_sh_y+avg_hip_y)
        vector_mid = (0, -1)
        angle = findAngle(vector_body[0], vector_body[1], vector_mid[0], vector_mid[1])
        # print(f'Angle: {angle}')
        if angle > 60:
            cv2.putText(img, 'FALL', (100, 120), cv2.FONT_HERSHEY_PLAIN, 10, red, 5)
    # if keypoints.pose_landmarks:
        mpDraw.draw_landmarks(img, keypoints.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.imshow('Video', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# import tensorflow as tf
# import tensorflow_hub as hub
# import cv2
# from matplotlib import pyplot as plt
# import numpy as np

# import tensorflow as tf
# import tensorflow_hub as hub
# import cv2
# import math as m
# from matplotlib import pyplot as plt
# import numpy as np
#
# # Optional if you are using a GPU
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
#
# model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
# movenet = model.signatures['serving_default']
#
# blue = (255, 127, 0)
# red = (50, 50, 255)
# green = (127, 255, 0)
# dark_blue = (127, 20, 0)
# light_green = (127, 233, 100)
# yellow = (0, 255, 255)
# pink = (255, 0, 255)
# def findAngle(x1, y1, x2, y2):
#     if x2 == x1 and y2 == y1:
#         return 90
#     if y1 == 0:
#         90
#     theta = m.acos(-(y2-y1)/(0.001+m.sqrt((x2-x1)**2+(y2-y1)**2)))
#     return int(180/m.pi)*theta
# def draw_connections(frame, keypoints, edges, confidence_threshold):
#     y, x, c = frame.shape
#     shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
#     a = 0
#     b = 0
#     for edge, color in edges.items():
#         p1, p2 = edge
#         y1, x1, c1 = shaped[p1]
#         y2, x2, c2 = shaped[p2]
#         if edge == (5, 6):
#             a = (x1+x2)/2
#             b = (y1+y2)/2
#             cv2.circle(frame, (int(a), int(b)), 6, blue, -1)
#         if edge == (11, 12):
#             a = a - (x1+x2)/2
#             b = b - (y1+y2)/2
#         if (c1 > confidence_threshold) & (c2 > confidence_threshold):
#             cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
#     return (a, b)
#
# def draw_keypoints(frame, keypoints, confidence_threshold):
#     y, x, c = frame.shape
#     shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
#
#     for kp in shaped:
#         ky, kx, kp_conf = kp
#         if kp_conf > confidence_threshold:
#             cv2.circle(frame, (int(kx), int(ky)), 6, (0, 255, 0), -1)
#
#
# # Function to loop through each person detected and render
# def loop_through_people(frame, keypoints_with_scores, edges, confidence_threshold):
#     for person in keypoints_with_scores:
#         vector_body = draw_connections(frame, person, edges, confidence_threshold)
#         draw_keypoints(frame, person, confidence_threshold)
#         if findAngle(vector_body[0], vector_body[1], 0, -1) > 60:
#             cv2.putText(frame, 'FALL', (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, red, 3)
# EDGES = {
#     (0, 1): 'm',
#     (0, 2): 'c',
#     (1, 3): 'm',
#     (2, 4): 'c',
#     (0, 5): 'm',
#     (0, 6): 'c',
#     (5, 7): 'm',
#     (7, 9): 'm',
#     (6, 8): 'c',
#     (8, 10): 'c',
#     (5, 6): 'y',
#     (5, 11): 'm',
#     (6, 12): 'c',
#     (11, 12): 'y',
#     (11, 13): 'm',
#     (13, 15): 'm',
#     (12, 14): 'c',
#     (14, 16): 'c'
# }
# filename = 'D:/video-fall-detection-master/queda.mp4'
# cap = cv2.VideoCapture(0)
# while cap.isOpened():
    # ret, frame = cap.read()
#
#     # Resize image
#     img = frame.copy()
#     img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 384, 640)
#     input_img = tf.cast(img, dtype=tf.int32)
#
#     # Detection section
#     results = movenet(input_img)
#     keypoints_with_scores = results['output_0'].numpy()[:, :, :51].reshape((6, 17, 3))
#
#     # Render keypoints
#     loop_through_people(frame, keypoints_with_scores, EDGES, 0.1)
#
#     cv2.imshow('Movenet Multipose', frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()