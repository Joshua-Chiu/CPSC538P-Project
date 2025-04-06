#!/usr/bin/env python3
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

# -----------------------------------------------------------------------------
# 1. Helper Functions & Model Definition
# -----------------------------------------------------------------------------

def load_pose(pose_data):
    """
    Convert pose data into a NumPy array.
    If pose_data is a list and its elements are Landmark objects (with attributes x, y, z),
    extract these values. Otherwise, assume pose_data is already numeric.
    Finally, if more than 33 keypoints are present, only the first 33 are returned.
    """
    if isinstance(pose_data, list):
        # Unwrap if needed
        if len(pose_data) > 0 and isinstance(pose_data[0], list):
            pose_data = pose_data[0]
        try:
            # Try to access attribute 'x'; if it works, convert each Landmark to [x, y, z]
            _ = pose_data[0].x
            pose = np.array([[kp.x, kp.y, kp.z] for kp in pose_data], dtype=np.float32)
        except AttributeError:
            # Otherwise, assume it's already a list of numbers/lists
            pose = np.array(pose_data, dtype=np.float32)
    elif isinstance(pose_data, np.ndarray):
        pose = pose_data.astype(np.float32)
    else:
        raise ValueError("Unknown pose data type")

    # If more than 33 keypoints are present, take only the first 33.
    if pose.shape[0] > 33:
        pose = pose[:33]
    return pose

class PoseEmbeddingNet(nn.Module):
    """
    Neural network converting input pose keypoints into an embedding.
    Expected input is a flat vector of length 33*3 = 99.
    Architecture:
      - Flatten input,
      - fc1: Linear(99, 512),
      - fc2: Linear(512, 256),
      - fc3: Linear(256, 128).
    """
    def __init__(self, input_dim=99, hidden_dim1=512, hidden_dim2=256, embedding_dim=128):
        super(PoseEmbeddingNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, embedding_dim)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input to (batch, 99)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# -----------------------------------------------------------------------------
# 2. Query Dataset Definition for "query_pose" Folder with Debugging
# -----------------------------------------------------------------------------

class QueryPoseDataset(Dataset):
    """
    Dataset for evaluation from the query folder.
    Each .pkl file is assumed to contain either a tuple (pose, label)
    or just pose data (in which case a dummy label -1 is assigned).
    The pose data is processed with load_pose() and converted to a tensor.
    """
    def __init__(self, folder_path):
        self.folder_path = folder_path
        abs_path = os.path.abspath(self.folder_path)
        print(f"[DEBUG] Absolute query folder path: {abs_path}")
        all_files = os.listdir(folder_path)
        print(f"[DEBUG] Files in folder: {all_files}")
        self.file_paths = [os.path.join(folder_path, f) for f in all_files if f.lower().endswith('.pkl')]
        print(f"[DEBUG] Filtered .pkl files: {self.file_paths}")
        print(f"[DEBUG] Found {len(self.file_paths)} .pkl files in {self.folder_path}")
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, index):
        file_path = self.file_paths[index]
        print(f"[DEBUG] Loading file: {file_path}")
        with open(file_path, "rb") as f:
            sample = pickle.load(f)
        print(f"[DEBUG] Loaded sample: {sample}")  # You can comment out this line if too verbose
        if isinstance(sample, tuple) and len(sample) == 2:
            pose, label = sample
        else:
            pose = sample
            label = -1  # assign dummy label
        pose = load_pose(pose)
        pose = torch.tensor(pose, dtype=torch.float32)
        return pose, label

# -----------------------------------------------------------------------------
# 3. Evaluation Functions (t-SNE & Nearest Neighbor Accuracy)
# -----------------------------------------------------------------------------

def plot_tsne(embeddings, labels, perplexity=30, n_iter=1000):
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
    nbrs = NearestNeighbors(n_neighbors=2, metric='cosine').fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)
    pred_labels = [labels[idx[1]] for idx in indices]
    acc = np.mean(np.array(pred_labels) == labels)
    return acc

# -----------------------------------------------------------------------------
# 4. Main Evaluation Script
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the trained model
    model = PoseEmbeddingNet(input_dim=99, hidden_dim1=512, hidden_dim2=256, embedding_dim=128).to(device)
    model.load_state_dict(torch.load("pose_embedding_model.pth", map_location=device))
    model.eval()
    print("✅ Model loaded for evaluation.")
    
    # Update the query folder path as needed
    query_folder = "/Users/rowelsabahat/Desktop/CPSC538P-Project/entireid/query_pose"
    query_dataset = QueryPoseDataset(query_folder)
    print("[DEBUG] Dataset length:", len(query_dataset))
    query_loader = DataLoader(query_dataset, batch_size=32, shuffle=False)
    
    # Compute embeddings for the query data
    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for batch_idx, (poses, labels) in enumerate(query_loader):
            print(f"[DEBUG] Processing batch {batch_idx+1} with {poses.size(0)} samples")
            poses = poses.to(device)
            embeddings = model(poses)
            print(f"[DEBUG] Batch embeddings shape: {embeddings.shape}")
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.extend(labels)
    
    if len(all_embeddings) == 0:
        print("[ERROR] No embeddings were collected. Please check the dataset and DataLoader.")
        exit(1)
    
    all_embeddings = np.vstack(all_embeddings)
    all_labels = np.array(all_labels)
    
    # t-SNE Visualization
    plot_tsne(all_embeddings, all_labels)
    
    # Nearest Neighbor Accuracy
    nn_acc = nearest_neighbor_accuracy(all_embeddings, all_labels)
    print("Query Nearest Neighbor Accuracy: {:.4f}".format(nn_acc))
