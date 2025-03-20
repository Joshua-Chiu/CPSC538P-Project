#!/usr/bin/env python3
import os
import glob
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

# -----------------------------------------------------------------------------
# 1. Helper Functions & Model Definition
# -----------------------------------------------------------------------------

def load_pose(pose_data):
    """
    Process pose data into a numpy array.
    This function assumes that pose_data is either:
      - A list of Landmark objects (with attributes x, y, z), or
      - Already a numpy array.
    """
    if isinstance(pose_data, list):
        pose = np.array([[kp.x, kp.y, kp.z] for kp in pose_data], dtype=np.float32)
    elif isinstance(pose_data, np.ndarray):
        pose = pose_data.astype(np.float32)
    else:
        raise ValueError("Unknown pose data type")
    return pose

class PoseEmbeddingNet(nn.Module):
    """
    Neural network that converts input pose keypoints into an embedding.
    Must match the architecture used during training.
    """
    def __init__(self, input_dim=34, embedding_dim=128):
        super(PoseEmbeddingNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )
        
    def forward(self, x):
        return self.fc(x)

# -----------------------------------------------------------------------------
# 2. Query Dataset Definition for "query_pose" Folder
# -----------------------------------------------------------------------------

class QueryPoseDataset(Dataset):
    """
    Dataset for evaluation from the "query_pose" folder.
    Assumes each .pkl file in the folder contains a tuple (pose, label).
    """
    def __init__(self, folder_path):
        self.folder_path = folder_path
        # List all .pkl files in the folder
        self.file_paths = glob.glob(os.path.join(folder_path, "*.pkl"))
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, index):
        file_path = self.file_paths[index]
        with open(file_path, "rb") as f:
            # Adjust this if your pickle files have a different structure.
            sample = pickle.load(f)
        # Expecting sample to be a tuple: (pose, label)
        pose, label = sample
        # Process the pose data
        pose = load_pose(pose)
        # Convert to tensor (ensure shape is as expected by the model, e.g., a flat vector of length 34)
        pose = torch.tensor(pose, dtype=torch.float32)
        return pose, label

# -----------------------------------------------------------------------------
# 3. Evaluation Functions
# -----------------------------------------------------------------------------

def compute_similarity_scores(embeddings, labels, metric='cosine'):
    """
    Compute similarity scores between each unique pair of embeddings.
    Returns:
      - scores: list of similarity scores.
      - gt: ground truth (1 if same label, 0 otherwise).
    """
    n = embeddings.shape[0]
    scores = []
    gt = []
    for i in range(n):
        for j in range(i + 1, n):
            if metric == 'cosine':
                norm_i = np.linalg.norm(embeddings[i])
                norm_j = np.linalg.norm(embeddings[j])
                sim = np.dot(embeddings[i], embeddings[j]) / (norm_i * norm_j)
            elif metric == 'euclidean':
                sim = -np.linalg.norm(embeddings[i] - embeddings[j])
            else:
                raise ValueError("Unsupported metric. Use 'cosine' or 'euclidean'.")
            scores.append(sim)
            gt.append(1 if labels[i] == labels[j] else 0)
    return np.array(scores), np.array(gt)

def compute_roc_auc(embeddings, labels, metric='cosine'):
    """
    Compute ROC-AUC score from pairwise similarity scores.
    """
    scores, gt = compute_similarity_scores(embeddings, labels, metric)
    auc = roc_auc_score(gt, scores)
    return auc

def plot_roc_curve(embeddings, labels, metric='cosine'):
    """
    Plot the ROC curve using pairwise similarity scores.
    """
    scores, gt = compute_similarity_scores(embeddings, labels, metric)
    fpr, tpr, thresholds = roc_curve(gt, scores)
    auc = roc_auc_score(gt, scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc='lower right')
    plt.show()

def plot_tsne(embeddings, labels, perplexity=30, n_iter=1000):
    """
    Perform t-SNE dimensionality reduction and visualize the embeddings.
    """
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    X_tsne = tsne.fit_transform(embeddings)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis')
    plt.title("t-SNE Visualization of Query Embeddings")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.colorbar(scatter, label='Label')
    plt.show()

def nearest_neighbor_accuracy(embeddings, labels):
    """
    Compute nearest neighbor accuracy by comparing each embedding's label
    to that of its closest neighbor (ignoring itself).
    """
    nbrs = NearestNeighbors(n_neighbors=2, metric='cosine').fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)
    # The first neighbor is the sample itself; use the second.
    pred_labels = [labels[idx[1]] for idx in indices]
    acc = np.mean(np.array(pred_labels) == labels)
    return acc

# -----------------------------------------------------------------------------
# 4. Main Evaluation Script
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Load the Trained Model ---
    model = PoseEmbeddingNet().to(device)
    model.load_state_dict(torch.load("pose_triplet_model.pth", map_location=device))
    model.eval()
    print("✅ Model loaded for evaluation.")
    
    # --- Load the Query Dataset ---
    # Change "query_pose" to the path of your query folder containing .pkl files.
    query_folder = "query_pose"
    query_dataset = QueryPoseDataset(query_folder)
    query_loader = DataLoader(query_dataset, batch_size=32, shuffle=False)
    
    # --- Compute Embeddings for the Query Data ---
    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for poses, labels in query_loader:
            poses = poses.to(device)
            embeddings = model(poses)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.extend(labels)
    all_embeddings = np.vstack(all_embeddings)
    all_labels = np.array(all_labels)
    
    # --- Evaluate the Model ---
    # 1. ROC-AUC Analysis (Print the score)
    auc = compute_roc_auc(all_embeddings, all_labels, metric='cosine')
    print("Query ROC-AUC Score: {:.4f}".format(auc))
    
    # 2. Plot the ROC Curve
    plot_roc_curve(all_embeddings, all_labels, metric='cosine')
    
    # 3. t-SNE Visualization
    plot_tsne(all_embeddings, all_labels)
    
    # 4. Nearest Neighbor Accuracy
    nn_acc = nearest_neighbor_accuracy(all_embeddings, all_labels)
    print("Query Nearest Neighbor Accuracy: {:.4f}".format(nn_acc))
