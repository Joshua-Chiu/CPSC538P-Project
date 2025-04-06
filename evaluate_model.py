#!/usr/bin/env python3
import os
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
# 1. Helper Functions & Updated Model Definition
# -----------------------------------------------------------------------------

def load_pose(pose_data):
    """
    Process pose data into a numpy array.
    This function assumes that pose_data is either:
      - A list of Landmark objects (with attributes x, y, z), or
      - Already a numpy array.
    Adjust this if your test pkl files have a different structure.
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
    This version is updated to match the training architecture:
      - fc1: Linear(99, 512)
      - fc2: Linear(512, 256)
      - fc3: Linear(256, 128)
    """
    def __init__(self, input_dim=99, hidden_dim1=512, hidden_dim2=256, embedding_dim=128):
        super(PoseEmbeddingNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, embedding_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# -----------------------------------------------------------------------------
# 2. Query Dataset Definition for "query_pose" Folder with Debugging
# -----------------------------------------------------------------------------

class QueryPoseDataset(Dataset):
    """
    Dataset for evaluation from the "query_pose" folder.
    Assumes each file in the folder is a .pkl file containing a tuple (pose, label).
    The pose data is expected to be compatible with the model input,
    i.e., a flat vector of length 99.
    """
    def __init__(self, folder_path):
        self.folder_path = folder_path
        abs_path = os.path.abspath(self.folder_path)
        print(f"[DEBUG] Absolute query folder path: {abs_path}")
        
        # List all files in the folder using os.listdir and filter for .pkl files (case-insensitive)
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
            # Expect each file to contain a tuple: (pose, label)
            sample = pickle.load(f)
            print(f"[DEBUG] Loaded sample type: {type(sample)}")
            print(f"[DEBUG] Sample contents: {sample}")

            if isinstance(sample, tuple) and len(sample) == 2:
                pose, label = sample
            else:
                raise ValueError(f"Invalid sample format in file {file_path}. Expected (pose, label), got: {sample}")

        pose = load_pose(pose)
        # Convert pose to tensor; expected shape is [99]
        pose = torch.tensor(pose, dtype=torch.float32).view(-1)  # shape becomes [99]

        return pose, label

# -----------------------------------------------------------------------------
# 3. Evaluation Functions
# -----------------------------------------------------------------------------

def compute_similarity_scores(embeddings, labels, metric='cosine'):
    """
    Compute similarity scores between each unique pair of embeddings.
    Returns:
      - scores: list of similarity scores.
      - gt: ground truth list (1 if same label, 0 otherwise).
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
    # Check if there is more than one unique class in labels
    if len(np.unique(labels)) < 2:
        print("[WARNING] Only one class found in labels. Skipping ROC AUC computation.")
        return None  # Return None if only one class is present
    
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
# 4. Main Evaluation Script with Debugging in Evaluation Loop
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Load the Trained Model ---
    model = PoseEmbeddingNet(input_dim=99, hidden_dim1=512, hidden_dim2=256, embedding_dim=128).to(device)
    model.load_state_dict(torch.load("pose_embedding_model.pth", map_location=device))
    model.eval()
    print("✅ Model loaded for evaluation.")
    
    # --- Load the Query Dataset ---
    # Update the path to the correct location if necessary
    query_folder = "/Users/andre/Desktop/CPSC538P-Project/entireid/query_pose"
    query_dataset = QueryPoseDataset(query_folder)
    print("[DEBUG] Dataset length:", len(query_dataset))
    query_loader = DataLoader(query_dataset, batch_size=32, shuffle=False)
    
    # --- Compute Embeddings for the Query Data ---
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
    
# --- Evaluate the Model ---

# Setup. Check unique classes in all_labels

unique_classes = np.unique(all_labels)
print(f"Unique classes in labels: {unique_classes}")
print(f"Number of unique classes: {len(unique_classes)}")
# 1. ROC-AUC Analysis (Print the score)
auc = compute_roc_auc(all_embeddings, all_labels, metric='cosine')

if auc is not None:
    print("Query ROC-AUC Score: {:.4f}".format(auc))
    plot_roc_curve(all_embeddings, all_labels, metric='cosine')
else:
    print("[INFO] Skipping ROC Curve plot due to insufficient class diversity.")

# 2. t-SNE Visualization
# After collecting all embeddings and labels in the loop
all_embeddings = np.vstack(all_embeddings)  # Converts list of arrays into a 2D NumPy array
all_labels = np.array(all_labels)

plot_tsne(all_embeddings, all_labels)

# 3. Nearest Neighbor Accuracy
nn_acc = nearest_neighbor_accuracy(all_embeddings, all_labels)
print("Query Nearest Neighbor Accuracy: {:.4f}".format(nn_acc))

