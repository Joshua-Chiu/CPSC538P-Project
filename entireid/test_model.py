import torch
import os
import random
import cv2
import mediapipe as mp
import numpy as np
from model import PoseEmbeddingNet  # Import the model you created

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2)

# Load the trained model
model = PoseEmbeddingNet(input_size=99, embedding_size=128)
model.load_state_dict(torch.load('pose_embedding_model.pth'))
model.eval()  # Set the model to evaluation mode

# Define a function to extract pose landmarks from an image
def extract_pose_landmarks(image_path):
    # Read the image
    image = cv2.imread(image_path)
    # Convert the image to RGB (MediaPipe expects RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Process the image to get pose landmarks
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        # Extract the x, y, z coordinates of the landmarks (33 landmarks with 3 values each)
        landmarks = [(landmark.x, landmark.y, landmark.z) for landmark in results.pose_landmarks.landmark]
        # Convert to a numpy array for easier manipulation
        return np.array(landmarks).flatten()  # Flatten to a 1D array (99 values)
    else:
        print(f"No pose landmarks detected in {image_path}")
        return None

# Function to calculate cosine similarity between two embeddings
def cosine_similarity(embedding1, embedding2):
    sim = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    return sim

# Randomly pick two images from the folder
folder_path = 'entireid/bounding_box_test'
images = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]

# Randomly select two image paths
image1_path = os.path.join(folder_path, random.choice(images))
image2_path = os.path.join(folder_path, random.choice(images))

# Print the names of the selected images
print(f"Comparing images: {os.path.basename(image1_path)} and {os.path.basename(image2_path)}")

# Extract pose landmarks for both images
landmarks1 = extract_pose_landmarks(image1_path)
landmarks2 = extract_pose_landmarks(image2_path)

# Check if landmarks were detected for both images
if landmarks1 is not None and landmarks2 is not None:
    # Convert landmarks to tensor
    landmarks1_tensor = torch.tensor(landmarks1, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    landmarks2_tensor = torch.tensor(landmarks2, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    
    # Get the embeddings (feature vectors) from the model
    with torch.no_grad():
        embedding1 = model(landmarks1_tensor).numpy()  # Convert to numpy for similarity calculation
        embedding2 = model(landmarks2_tensor).numpy()

    # Compute cosine similarity between the embeddings
    similarity = cosine_similarity(embedding1, embedding2)
    print(f"Cosine similarity: {similarity}")

    # Set a threshold for the similarity to decide if they are the same person
    threshold = 0.8  # You can adjust this threshold based on your experiments
    if similarity > threshold:
        print("The images are of the same person.")
    else:
        print("The images are of different people.")
else:
    print("Pose landmarks were not detected for one or both images.")
