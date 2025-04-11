import os
import torch
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from train_triplets import PoseEmbeddingNet
import mediapipe as mp
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load your trained model
model = PoseEmbeddingNet()
model.load_state_dict(torch.load('pose_embedding_model.pth'))
model.eval()

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Transform for images (not used here unless feeding raw images)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_pose_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    return results.pose_landmarks if results.pose_landmarks else None

def convert_landmarks_to_tensor(landmarks):
    landmarks_list = [[lm.x, lm.y, lm.z] for lm in landmarks.landmark]
    return torch.tensor(landmarks_list).float().unsqueeze(0)

def extract_embeddings_from_dataset(dataset_path):
    embeddings, image_paths = [], []

    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith((".png", ".jpg")):
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path)
                landmarks = extract_pose_landmarks(image)
                if landmarks:
                    landmarks_tensor = convert_landmarks_to_tensor(landmarks)
                    with torch.no_grad():
                        embedding = model(landmarks_tensor)
                    embeddings.append(embedding.squeeze().cpu().numpy())
                    image_paths.append(image_path)

    return np.array(embeddings), image_paths

def scale_embeddings(embeddings):
    return StandardScaler().fit_transform(embeddings)

def reduce_dimensions_with_pca(embeddings, n_components=50):
    return PCA(n_components=n_components).fit_transform(embeddings)

def cluster_embeddings(embeddings):
    clustering = DBSCAN(eps=0.2, min_samples=3, metric='euclidean').fit(embeddings)
    return clustering.labels_

def evaluate_cluster_count(labels):
    unique_labels = set(labels)
    unique_labels.discard(-1)  # Remove noise
    return len(unique_labels), unique_labels

def visualize_embeddings(embeddings, labels):
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    labels = np.array(labels)
    unique_labels = np.unique(labels)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        reduced_embeddings[:, 0], 
        reduced_embeddings[:, 1], 
        c=labels, 
        cmap='tab20', 
        s=15,
        alpha=0.8
    )

    cbar = plt.colorbar(scatter, ticks=range(min(labels), max(labels)+1))
    cbar.set_label("Cluster ID")
    cbar.set_ticks(unique_labels)
    
    plt.title("t-SNE visualization of embeddings with clusters")
    plt.xlabel("t-SNE component 1")
    plt.ylabel("t-SNE component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_ETHZ", "seq2")
    
    embeddings, image_paths = extract_embeddings_from_dataset(dataset_path)
    print(f"Extracted {len(embeddings)} embeddings.")

    embeddings_scaled = scale_embeddings(embeddings)
    embeddings_pca = reduce_dimensions_with_pca(embeddings_scaled, n_components=50)

    labels = cluster_embeddings(embeddings_pca)
    labels = labels.astype(int)  # Ensure they are integers for plotting

    num_clusters, unique_labels = evaluate_cluster_count(labels)
    print(f"Number of unique individuals detected: {num_clusters}")
    print(f"Unique label IDs: {unique_labels}")

    # ✅ Compute silhouette score
    if len(set(labels)) > 1 and not all(label == -1 for label in labels):
        sil_score = silhouette_score(embeddings_pca, labels)
        print(f"Silhouette Score: {sil_score:.4f}")
    else:
        sil_score = None
        print("Silhouette Score: Cannot compute (need at least 2 clusters excluding noise).")

    visualize_embeddings(embeddings_scaled, labels)

    with open("clustering_results.txt", "w") as f:
        f.write(f"Unique individuals: {num_clusters}\n")
        f.write(f"Unique label IDs: {unique_labels}\n")
        if sil_score is not None:
            f.write(f"Silhouette Score: {sil_score:.4f}\n")
        else:
            f.write("Silhouette Score: Not available (need at least 2 clusters).\n")
