import os
import torch
import pickle
import numpy as np
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from unsupervised_learning_ import PoseAutoencoder

# ---------------------- Load the trained model ---------------------- #
def load_model(model_path="pose_autoencoder.pth"):
    model = PoseAutoencoder()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# ---------------------- MediaPipe Pose Extraction ---------------------- #
def extract_pose_landmarks(image_path):
    # Initialize MediaPipe Pose model
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9)
    
    # Load image using OpenCV
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image and extract pose landmarks
    results = pose.process(image_rgb)
    
    # Extract landmarks if they exist
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
        return landmarks_array
    return None  # Return None if no landmarks are found

# ---------------------- Extract embeddings from images ---------------------- #
def extract_embeddings_from_images(image_folder, model):
    embeddings = []
    image_paths = []
    
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        landmarks = extract_pose_landmarks(image_path)
        
        if landmarks is not None and len(landmarks) == 33:
            # Flatten the landmarks and create the tensor
            flattened_landmarks = landmarks.flatten()
            sample_tensor = torch.tensor(flattened_landmarks)
            
            # Pass through the encoder to get the embedding (z)
            with torch.no_grad():
                _, embedding = model(sample_tensor)
            
            embeddings.append(embedding.numpy())  # Store the embedding as numpy array
            image_paths.append(image_path)  # Optionally store the image paths for later reference
    
    return np.array(embeddings), image_paths

# ---------------------- Apply t-SNE and KMeans ---------------------- #
def apply_tsne_and_kmeans(embeddings, image_paths, n_clusters=None):
    # t-SNE for dimensionality reduction to 2D
    tsne = TSNE(n_components=2, random_state=42)
    tsne_embeddings = tsne.fit_transform(embeddings)
    
    # Apply KMeans clustering to group similar embeddings
    if n_clusters is None:
        n_clusters = len(image_paths)  # If labels are not provided, assume one cluster per image
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    
    return tsne_embeddings, clusters

# ---------------------- Plot t-SNE with Clusters ---------------------- #
def plot_tsne(tsne_embeddings, clusters, image_paths):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=clusters, cmap='viridis', s=50, alpha=0.7)
    plt.title("t-SNE Visualization of Pose Embeddings")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    
    # Create legend for clusters
    handles, _ = scatter.legend_elements()
    plt.legend(handles, [f"Cluster {i}" for i in range(len(set(clusters)))], title="Clusters")
    
    # Optionally print the image paths for reference
    for i, txt in enumerate(image_paths):
        plt.annotate(txt, (tsne_embeddings[i, 0], tsne_embeddings[i, 1]), fontsize=8, alpha=0.6)

    plt.show()

# ---------------------- Main ---------------------- #
if __name__ == "__main__":
    # Load the trained model
    model = load_model("pose_autoencoder.pth")
    
    # Extract embeddings from a new dataset of PNG images
    embeddings, image_paths = extract_embeddings_from_images("dataset_ETHZ/seq2", model)
    
    # Apply t-SNE and KMeans
    tsne_embeddings, clusters = apply_tsne_and_kmeans(embeddings, image_paths)
    
    # Plot the t-SNE visualization with clusters
    plot_tsne(tsne_embeddings, clusters, image_paths)
