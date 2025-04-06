import os
import cv2
import torch
import mediapipe as mp
from sklearn.metrics import roc_curve, auc
import numpy as np
from train_triplets import PoseEmbeddingNet  # Import your model definition file

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize pose embedding model
pose_embedding_model = PoseEmbeddingNet(input_size=99, embedding_size=128)  # Adjust according to your model
pose_embedding_model.load_state_dict(torch.load('pose_embedding_model.pth'))
pose_embedding_model.eval()  # Set the model to evaluation mode

def extract_pose_landmarks(image_path):
    image = cv2.imread(image_path)
    
    # Check if the image is loaded properly
    if image is None:
        print(f"Error loading image: {image_path}")
        return None  # Skip processing if the image couldn't be loaded

    # Convert the image to RGB before passing it to MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and extract pose landmarks
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z])
        return torch.tensor(landmarks).flatten()  # Flatten the landmarks to 1D tensor
    else:
        print(f"No pose landmarks detected in {image_path}")
        return None

def evaluate_model(image_pairs):
    true_labels = []
    predicted_scores = []

    for img1_path, img2_path, label in image_pairs:
        # Extract pose landmarks from both images
        landmarks1 = extract_pose_landmarks(img1_path)
        landmarks2 = extract_pose_landmarks(img2_path)

        # Skip pairs where landmarks could not be extracted
        if landmarks1 is None or landmarks2 is None:
            continue

        # Get embeddings for both images using the pose embedding model
        embedding1 = pose_embedding_model(landmarks1)
        embedding2 = pose_embedding_model(landmarks2)

        # Compute the similarity between embeddings (e.g., using cosine similarity)
        similarity = torch.nn.functional.cos
