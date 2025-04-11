import os
import cv2
import pickle
import mediapipe as mp
from tqdm import tqdm

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.7)

# Input and output paths
dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "entireid", "bounding_box_test")  # CHANGE THIS
output_pkl = "pose_landmarks_unsupervised_dataset.pkl"

pose_data = []

# Loop through all images
for root, _, files in os.walk(dataset_path):
    for file in tqdm(files):
        if file.lower().endswith((".png", ".jpg")):
            image_path = os.path.join(root, file)
            image = cv2.imread(image_path)

            if image is None:
                continue  # skip corrupted or unreadable files

            # Convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                landmarks = [
                    [lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark
                ]
                pose_data.append({
                    "image_path": image_path,
                    "landmarks": landmarks
                })

# Save as .pkl
with open(output_pkl, "wb") as f:
    pickle.dump(pose_data, f)

print(f"✅ Saved {len(pose_data)} landmark entries to {output_pkl}")
