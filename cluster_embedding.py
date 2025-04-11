import os
import torch
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from train_triplets import PoseEmbeddingNet  # Replace with the actual import path for your model
import mediapipe as mp
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load your trained model
model = PoseEmbeddingNet()  # Replace with your actual model class
model.load_state_dict(torch.load('pose_embedding_model.pth'))  # Adjust the path accordingly
model.eval()  # Set the model to evaluation mode

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Transformations for input images to tensor (if needed)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to extract pose landmarks from an image
def extract_pose_landmarks(image):
    # Convert the image to RGB for MediaPipe processing
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        return results.pose_landmarks
    return None

# Function to convert MediaPipe landmarks to a tensor
def convert_landmarks_to_tensor(landmarks):
    # Extract x, y, z coordinates from the landmarks
    landmarks_list = []
    for landmark in landmarks.landmark:
        landmarks_list.append([landmark.x, landmark.y, landmark.z])
    
    # Convert to a tensor
    landmarks_tensor = torch.tensor(landmarks_list).float().unsqueeze(0)  # Add batch dimension
    return landmarks_tensor

# Function to extract embeddings from the dataset
def extract_embeddings_from_dataset(dataset_path):
    embeddings = []
    image_paths = []

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".png") or file.endswith(".jpg"):
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path)

                # Extract pose landmarks
                landmarks = extract_pose_landmarks(image)
                if landmarks:
                    # Convert landmarks to tensor and pass through the model to get embeddings
                    landmarks_tensor = convert_landmarks_to_tensor(landmarks)
                    with torch.no_grad():
                        embedding = model(landmarks_tensor)  # Pass landmarks to the model
                    embeddings.append(embedding.squeeze().cpu().numpy())  # Store embeddings
                    image_paths.append(image_path)  # Optionally keep track of image paths

    return np.array(embeddings), image_paths

# Function to cluster the embeddings
def cluster_embeddings(embeddings):
    # Use DBSCAN to cluster embeddings
    clustering = DBSCAN(eps=0.2, min_samples=3, metric='euclidean').fit(embeddings)
    return clustering.labels_  # Cluster labels for each image

# Evaluate the number of unique individuals in the dataset
def evaluate_cluster_count(labels):
    unique_labels = set(labels)
    # Exclude noise points (-1 label in DBSCAN)
    unique_labels.discard(-1)
    return len(unique_labels), unique_labels

# Reduce dimensions of embeddings for visualization
def visualize_embeddings(embeddings, labels):
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    # Plotting
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='tab20', s=10)
    plt.title("t-SNE visualization of embeddings")
    plt.colorbar(scatter, label='Cluster ID')
    plt.xlabel("t-SNE component 1")
    plt.ylabel("t-SNE component 2")
    plt.show()

if __name__ == "__main__":
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_ETHZ", "seq2")  # Adjust path
    embeddings, image_paths = extract_embeddings_from_dataset(dataset_path)

    print(f"Extracted {len(embeddings)} embeddings.")

    # Perform clustering
    labels = cluster_embeddings(embeddings)

    # Evaluate number of unique individuals
    num_unique_individuals, unique_labels = evaluate_cluster_count(labels)
    
    print(f"Number of unique individuals detected: {num_unique_individuals}")
    print(f"Unique label IDs: {unique_labels}")

    # Visualize the clusters with t-SNE
    visualize_embeddings(embeddings, labels)

    # Optionally save or output results
    with open("clustering_results.txt", "w") as f:
        f.write(f"Unique individuals: {num_unique_individuals}\n")
        f.write(f"Unique label IDs: {unique_labels}\n")
