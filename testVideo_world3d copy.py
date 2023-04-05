import cv2
import numpy as np
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
        # cv2.putText(annotated_image, f"{handedness[0].category_name}",
        #             (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
        #             FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image

def plot(ax,plt,data,xlim=(-0.5, 0.1),ylim=(-0.5, 0.1),zlim=(0.2, 1.0),autoscale = False):
        # Create 3D plot

        if data.shape >= (21,3):
          
            # Clear the plot and add new data
            ax.clear()
            
            # auto scale the plot
            if autoscale:
                ax.autoscale(enable=True, axis='both', tight=None)
            else:
                ax.set_xlim3d(xlim)
                ax.set_ylim3d(ylim)
                ax.set_zlim3d(zlim)
                ax.scatter3D(*zip(*data))
     
            #  C
            edges = [(1,2),(2,3),(3,4),(0,5),(5,6),(5,9),(1,0),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]
            edges2 = [(22,23),(23,24),(24,25),(21,26),(26,27),(26,30),(22,21),(27,28),(28,29),(21,30),(30,31),(31,32),(32,33),(30,34),(34,35),(35,36),(36,37),(34,38),(38,39),(39,40),(40,41),(21,38)] 

            if data.shape != (42,3):
                for edge in edges:
                    ax.plot3D(*zip(data[edge[0]], data[edge[1]]), color='red')

            else:
                for edge in edges:
                    ax.plot3D(*zip(data[edge[0]], data[edge[1]]), color='red')
                for edge in edges2:
                    ax.plot3D(*zip(data[edge[0]], data[edge[1]]), color='blue')


            # Draw the plot
            plt.draw()
            plt.pause(0.1)


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
        focal_length = 0.9*frame_width
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
        if i == 0:
            rvec = np.zeros(3)
            tvec = np.asarray([0,0,0.6])
        success, rvec, tvec, = cv2.solvePnP(
                    model_points,
                    image_points,
                    camera_matrix,
                    distortion,
                    rvec,
                    tvec,
                    useExtrinsicGuess = True, 
                    flags=cv2.SOLVEPNP_SQPNP
                )        
        # ######################## METHOD 1 ########################
        # R, _ = cv2.Rodrigues(rvec)
        # transformation = np.eye(4, dtype=np.float32)
        # transformation[:3,:3] = R
        # transformation[:3,3] = tvec.squeeze()
        # model_points_hom = np.concatenate((model_points, np.ones((21, 1))), axis=1)
        # world_points_in_camera = (model_points_hom @ transformation.T)[:,0:3]
        # # world_points_in_camera = transformation @ model_points_hom.T
        # # world_points_in_camera = world_points_in_camera[0:3,:].T
        # # world_points_in_camera = model_points @ R.T + tvec.T
        # world_points_in_camera_z = world_points_in_camera[:, 2].reshape(-1, 1)
        # world_points_in_camera[:, :2] =  (np.concatenate([image_points, np.ones((image_points.shape[0], 1))], axis=1) @ np.linalg.inv(camera_matrix).T * world_points_in_camera_z)[:, :2]
        # ######################## METHOD 1 ########################
        
        ####################### METHOD 2 ########################
        R, _ = cv2.Rodrigues(rvec)
        transformation = np.eye(4, dtype=np.float32)
        transformation[:3,3] = tvec.squeeze()
        model_points_hom = np.concatenate((model_points, np.ones((21, 1))), axis=1)
        world_points_in_camera = model_points_hom.dot(np.linalg.inv(transformation).T)
        # world_points_in_camera_z = world_points_in_camera[:, 2].reshape(-1, 1)
        # world_points_in_camera[:, :2] =  (np.concatenate([image_points, np.ones((image_points.shape[0], 1))], axis=1) @ np.linalg.inv(camera_matrix).T * world_points_in_camera_z)[:, :2]
        ####################### METHOD 2 ########################

        # ####################### METHOD 3 ########################
        # world_points_in_camera = model_points + tvec
        # world_points_in_camera_z = world_points_in_camera[:, 2].reshape(-1, 1)
        # world_points_in_camera[:, :2] =  (np.concatenate([image_points, np.ones((image_points.shape[0], 1))], axis=1) @ np.linalg.inv(camera_matrix).T * world_points_in_camera_z)[:, :2]
        # ####################### METHOD 3 ########################
        keypoints.append(world_points_in_camera)
        annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), hand_landmarker_result)
        out.write(cv2.cvtColor(annotated_image,cv2.COLOR_RGB2BGR))
        cv2.imshow('MediaPipe Hands', cv2.cvtColor(annotated_image,cv2.COLOR_RGB2BGR))
        # Press esc on keyboard to  exit
        if cv2.waitKey(5) & 0xFF == 27:
            break

savemat('./output3/data.mat',{'fps':fps,'timestampList':timestampList,'keypoints':keypoints})
cap.release()
out.release()


import matplotlib.pyplot as plt
fig = plt.figure()
plt.ion()
ax = fig.add_subplot(111, projection='3d')
for keypoint in keypoints:
    plot(ax,plt,keypoint,autoscale=True)