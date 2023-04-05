import cv2
import numpy as np
import pyzed.sl as sl
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from scipy.io import savemat

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
        cv2.putText(annotated_image, f"HAND {idx}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image

# import gi
# gi.require_version('Gtk', '2.0')
zed = sl.Camera()
init_params = sl.InitParameters()
# init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
init_params.coordinate_units = sl.UNIT.METER          # Set coordinate units (for depth)
init_params.depth_mode = sl.DEPTH_MODE.QUALITY
init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

name = "PUSH_PULL"
filepath = './zed_hand3/{}.svo'.format(name)
print("Using SVO file: {0}".format(filepath))
init_params.svo_real_time_mode = False
init_params.set_from_svo_file(filepath)
# Open the camera
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    exit(1)
# Get the intrinsic camera parameters of the ZED camera
fx, fy, cx, cy = zed.get_camera_information().calibration_parameters.left_cam.fx, \
    zed.get_camera_information().calibration_parameters.left_cam.fy, \
    zed.get_camera_information().calibration_parameters.left_cam.cx, \
    zed.get_camera_information().calibration_parameters.left_cam.cy
fps = zed.get_camera_information().camera_fps
Nframe = zed.get_svo_number_of_frames()
frame_width = zed.get_camera_information().camera_resolution.width
frame_height = zed.get_camera_information().camera_resolution.height

# Create and set RuntimeParameters after opening the camera
runtime_parameters = sl.RuntimeParameters()
# Use STANDARD sensing mode
runtime_parameters.sensing_mode = sl.SENSING_MODE.FILL # FILL NAN in point clouds.
# Setting the depth confidence parameters
runtime_parameters.confidence_threshold = 100
runtime_parameters.textureness_confidence_threshold = 100

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a hand landmarker instance with the video mode:
options = HandLandmarkerOptions(
    num_hands = 2,
    base_options=BaseOptions(model_asset_path='./hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO)
out = cv2.VideoWriter('./zed_hand3/{}.mp4'.format(name),cv2.VideoWriter_fourcc(*'MP4V'), fps, (int(frame_width),int(frame_height)))
timestampList = []
keypoints = []
with HandLandmarker.create_from_options(options) as landmarker:
    depth_image = sl.Mat()
    point_cloud = sl.Mat()
    color_image = sl.Mat()
    for i in range(Nframe):
        # Capture frames from the ZED video file
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_measure(depth_image, sl.MEASURE.DEPTH)
            zed.retrieve_measure(point_cloud,sl.MEASURE.XYZRGBA)
            zed.retrieve_image(color_image, sl.VIEW.LEFT)
            imgHeight = depth_image.get_height()
            imgWidth = depth_image.get_width()
            # channels = color_image.get_channels() # Note that there are 4 channels since ZED's grab is BGRA format.
            depth_image_np = depth_image.get_data()
            point_cloud_np = point_cloud.get_data()
            color_image_np_bgra = color_image.get_data()
            color_image_np_rgb = cv2.cvtColor(color_image_np_bgra, cv2.COLOR_BGRA2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,data=color_image_np_rgb)
            timestamp = int(i*1/fps*1000)
            timestampList.append(timestamp)
            hand_landmarker_result = landmarker.detect_for_video(mp_image, timestamp)
            print("{},{}".format(i,len(hand_landmarker_result.hand_landmarks)))
            hand_landmarks_list = hand_landmarker_result.hand_landmarks
            hand_landmarks = hand_landmarks_list[1] # Consider one hand
            hand_landmark_wrist = hand_landmarks[mp.solutions.hands.HandLandmark.WRIST]
            y_wrist = int(hand_landmark_wrist.y*imgHeight) 
            x_wrist = int(hand_landmark_wrist.x*imgWidth)
            depth_wrist = depth_image_np[y_wrist, x_wrist]
            print("Frame {}, depth_wrist:{}".format(i,depth_wrist))
            keypoint = []
            for landmark in hand_landmarks:
                u = int(landmark.x*imgWidth)
                v = int(landmark.y*imgHeight)
                # point = point_cloud.get_value(u,v)
                # point = point_cloud_np[v,u]
                # keypoint.append([point[0],point[1],point[2]])
                d = depth_wrist + landmark.z*depth_wrist
                keypoint.append([(u-cx)*d/fx,(v-cy)*d/fy,d])

            keypoints.append(keypoint)
            annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), hand_landmarker_result)
            out.write(cv2.cvtColor(annotated_image,cv2.COLOR_RGB2BGR))
            cv2.imshow('MediaPipe Hands', cv2.cvtColor(annotated_image,cv2.COLOR_RGB2BGR))
        else:
            Nframe = i
            print("frame {} not grabbed".format(i))
            break
        # Press esc on keyboard to  exit
        if cv2.waitKey(5) & 0xFF == 27:
                break
# Release resources
# print(len(timestampList))
# print(len(keypoints))
# print(len(keypoints[1]))
savemat('./zed_hand3/{}.mat'.format(name),
        {'fps': fps, 'timestampList': timestampList, 'keypoints': keypoints})
cv2.destroyAllWindows()
zed.close()
out.release()
