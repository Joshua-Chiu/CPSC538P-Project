import os
import cv2
import pickle

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

def save_object(object, path: str, overwrite=False) -> bool:
    print(f"save_object: saving object {path}")
    target_path = os.path.join(path)
    with open(target_path, 'wb') as f:
        pickle.dump(object, f)


def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image

def generate_coords(file_path):
    print(f"generate_coords: {file_path}")

    # Check the file exists and set path variables
    if not os.path.isfile(file_path):
        print("Error: The specified file does not exist.")
        return
    parent_dir = os.path.dirname(file_path)
    parent_name = os.path.basename(parent_dir)
    destination_folder = os.path.join(os.path.dirname(parent_dir), f"{parent_name}_pose")
    debug_folder = os.path.join(os.path.dirname(parent_dir), f"{parent_name}_overlay")
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    os.makedirs(destination_folder, exist_ok=True)
    os.makedirs(debug_folder, exist_ok=True)
    
    # Show the image and wait for a keypress
    img = cv2.imread(file_path)
    # cv2.imshow(f"{file_name}", img)
    # 

    base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
        )
    detector = vision.PoseLandmarker.create_from_options(options)

    image = mp.Image.create_from_file(file_path)
    detection_result = detector.detect(image)

    if detection_result.pose_world_landmarks:
        save_object(detection_result.pose_world_landmarks, os.path.join(destination_folder, f"{file_name}.pkl"))

    annotated_image = draw_landmarks_on_image(img, detection_result)
    # cv2.imshow(f"{file_name}",cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(debug_folder, f"{file_name}.png"), annotated_image)
    # cv2.waitKey(1)
    # cv2.destroyAllWindows()


    









PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(PATH, "entireid")

for item in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, item)
    
    # Check if the item is a directory (subfolder)
    if os.path.isdir(folder_path):
        # List files in the subfolder
        files = os.listdir(folder_path)
        for file in files:
            if file.lower().endswith(".jpg"):
                generate_coords(os.path.join(folder_path, file))
