import cv2
import numpy as np
import mediapipe as mp
# import matplotlib.pyplot as plt
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
from scipy.io import savemat
import os
from tqdm import tqdm, trange
from one_euro_filter import OneEuroFilter

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green


def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        # cv2.putText(annotated_image, f"{handedness[0].category_name}",
        #             (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
        #             FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image


# def depth_to_distance(depth):
#     # Decided by camera
#     return -1.7 * depth + 2


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a hand landmarker instance with the video mode:
base_options=BaseOptions(model_asset_path='./hand_landmarker.task')
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='./hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO)
options.num_hands = 2
model_path = os.path.abspath(
    "./model-f6b98070.onnx")  # MiDas v2.1 model large
# model_path = os.path.abspath("./model-small.onnx") # MiaDas v2.1 model small
model = cv2.dnn.readNet(model_path)

cap = cv2.VideoCapture("./videos/avatar.jpg213.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
out = cv2.VideoWriter('./output2/output.mp4', cv2.VideoWriter_fourcc(*
                      'MP4V'), fps, (int(frame_width), int(frame_height)))
out_depth = cv2.VideoWriter('./output2/output_depth.mp4', cv2.VideoWriter_fourcc(*
                                                                                 'MP4V'), fps, (int(frame_width), int(frame_height)), 0)
cx = frame_width*0.5
cy = frame_height*0.5
fx = frame_width*0.9
fy = frame_width*0.9
Nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)
timestampList = []
keypoints = []
smooth_keypoints = []
handworld_keypoints = []
print("fps=", fps, "frames=", Nframes)
with HandLandmarker.create_from_options(options) as landmarker:
    for i in trange(int(Nframes)):
        success, img = cap.read()  # frame: BGR
        if not success:
            break
        img = cv2.flip(img,1)      # Flip image for 3rd person view
        imgHeight, imgWidth, channels = img.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        frame_timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        timestampList.append(frame_timestamp_ms)
        hand_landmarker_result = landmarker.detect_for_video(
            mp_image, int(frame_timestamp_ms))
        hand_landmarks_list = hand_landmarker_result.hand_landmarks
        if len(hand_landmarks_list) >0:
            hand_landmarks = hand_landmarks_list[0]  # Consider one hand
        # -------------- Depth map from neural net ---------------------------
        # Create Blob from Input Image
        # MiDaS v2.1 Large ( Scale : 1 / 255, Size : 384 x 384, Mean Subtraction : ( 123.675, 116.28, 103.53 ), Channels Order : RGB )
        blob = cv2.dnn.blobFromImage(
            img, 1/255., (384, 384), (123.675, 116.28, 103.53), True, False)

        # MiDaS v2.1 Small ( Scale : 1 / 255, Size : 256 x 256, Mean Subtraction : ( 123.675, 116.28, 103.53 ), Channels Order : RGB )
        #blob = cv2.dnn.blobFromImage(img, 1/255., (256,256), (123.675, 116.28, 103.53), True, False)
        model.setInput(blob)
        depth_map = model.forward()
        depth_map = depth_map[0, :, :]
        depth_map = cv2.resize(depth_map, (imgWidth, imgHeight))
        # Normalize the output
        depth_map = cv2.normalize(
            depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        hand_landmark_wrist = hand_landmarks[mp.solutions.hands.HandLandmark.WRIST]
        y_wrist = int(hand_landmark_wrist.y*imgHeight)
        x_wrist = int(hand_landmark_wrist.x*imgWidth)
        depth_wrist = depth_map[y_wrist, x_wrist]
        keypoint = []
        for landmark in hand_landmarks:
                u = int(landmark.x*imgWidth)
                v = int(landmark.y*imgHeight)
                d = depth_wrist + landmark.z*depth_wrist
                keypoint.append([(u-cx)*d/fx,(v-cy)*d/fy,d])
        keypoint = np.array(keypoint)
        hand_world_landmarks_list = hand_landmarker_result.hand_world_landmarks
        if len(hand_world_landmarks_list)>0:
            hand_world_landmarks = hand_world_landmarks_list[0]
        model_points = np.float32([[-l.x, -l.y, -l.z] for l in hand_world_landmarks])
        handworld_keypoints.append(model_points)
        # Resize hand to 0.1 meter palm length
        keypoint = keypoint*0.1/np.linalg.norm(keypoint[0,:]-keypoint[9,:]) 
        if i == 0:
            # Initialize the one-euro filter with suitable parameters
            filter = OneEuroFilter(x0=keypoint, dx0=0.0, t0 = 0,min_cutoff=0.004, beta=0.7, d_cutoff=1.0)
            smooth_keypoints.append(keypoint)
        else:
            smooth_keypoint = filter(keypoint, t = frame_timestamp_ms*1000)
            smooth_keypoints.append(smooth_keypoint)
        keypoints.append(keypoint)
        annotated_image = draw_landmarks_on_image(
            mp_image.numpy_view(), hand_landmarker_result)
        out.write(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        out_depth.write((depth_map*255).astype(np.uint8))
        # cv2.imshow('MediaPipe Hands', cv2.cvtColor(
        #     annotated_image, cv2.COLOR_RGB2BGR))
        # cv2.imshow('Depth map', depth_map)
        # Press esc on keyboard to  exit
        if cv2.waitKey(5) & 0xFF == 27:
            break
# print(len(timestampList))
# print(len(keypoints))
# print(len(keypoints[1]))
savemat('./output2/data.mat',
        {'fps': fps, 'timestampList': timestampList, 'keypoints': keypoints,'smooth_keypoints':smooth_keypoints,'handworld_keypoints':handworld_keypoints})
cap.release()
out.release()
