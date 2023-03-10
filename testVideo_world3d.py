import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
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
        # cv2.putText(annotated_image, f"{handedness[0].category_name}",
        #             (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
        #             FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a hand landmarker instance with the video mode:
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='./hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO)
cap = cv2.VideoCapture("./videos/3.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
out = cv2.VideoWriter('./output3/output.mp4',cv2.VideoWriter_fourcc(*'MP4V'), fps, (int(frame_width),int(frame_height)))
Nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)
timestampList = []
keypoints = []
print("fps=", fps, "frames=", Nframes)
with HandLandmarker.create_from_options(options) as landmarker:
    for i in range(int(Nframes)):
        success, img = cap.read()  # frame: BGR
        if not success:
            break
        frame_height, frame_width, channels = img.shape
        # https://github.com/google/mediapipe/issues/2199
        # https://gist.github.com/eldog/9012ce957be26934044131daffc25c73
        focal_length = frame_width
        center = (frame_width/2, frame_height/2)
        camera_matrix = np.array(
                                 [[focal_length, 0, center[0]],
                                 [0, focal_length, center[1]],
                                 [0, 0, 1]], dtype = "double"
                                 )
        distortion = np.zeros((4, 1))


        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        frame_timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        timestampList.append(frame_timestamp_ms)
        hand_landmarker_result = landmarker.detect_for_video(mp_image, int(frame_timestamp_ms))
        hand_landmarks_list = hand_landmarker_result.hand_landmarks
        hand_world_landmarks_list = hand_landmarker_result.hand_world_landmarks
        hand_landmarks = hand_landmarks_list[0] # Consider one hand
        hand_world_landmarks = hand_world_landmarks_list[0]
        model_points = np.float32([[-l.x, -l.y, -l.z] for l in hand_world_landmarks])
        image_points = np.float32([[l.x * frame_width, l.y * frame_height] for l in hand_landmarks])
        success, rvecs, tvecs, = cv2.solvePnP(
                    model_points,
                    image_points,
                    camera_matrix,
                    distortion, 
                    flags=cv2.SOLVEPNP_SQPNP
                )
        transformation = np.eye(4)  # needs to 4x4 because you have to use homogeneous coordinates
        transformation[0:3, 3] = tvecs.squeeze()
        # the transformation consists only of the translation, because the rotation is accounted for in the model coordinates. Take a look at this (https://codepen.io/mediapipe/pen/RwGWYJw to see how the model coordinates behave - the hand rotates, but doesn't translate

        # transform model coordinates into homogeneous coordinates
        model_points_hom = np.concatenate((model_points, np.ones((21, 1))), axis=1)

        # apply the transformation
        world_points = model_points_hom.dot(np.linalg.inv(transformation).T)
        keypoints.append(world_points[:,0:3])


        annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), hand_landmarker_result)
        out.write(cv2.cvtColor(annotated_image,cv2.COLOR_RGB2BGR))
        cv2.imshow('MediaPipe Hands', cv2.cvtColor(annotated_image,cv2.COLOR_RGB2BGR))
        # Press esc on keyboard to  exit
        if cv2.waitKey(5) & 0xFF == 27:
            break
savemat('./output3/data.mat',{'fps':fps,'timestampList':timestampList,'keypoints':keypoints})
cap.release()
out.release()