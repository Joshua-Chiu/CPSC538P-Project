import os
import torch
import numpy as np
import cv2
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from train_triplets import PoseEmbeddingNet
import mediapipe as mp
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform

# Load your trained model - this is your pose embedding network
model = PoseEmbeddingNet()
model.load_state_dict(torch.load('pose_embedding_model.pth'))
model.eval()

# Initialize MediaPipe Pose for extracting landmarks
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

def extract_pose_landmarks(image):
    """Extract pose landmarks from an image using MediaPipe"""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    return results.pose_landmarks if results.pose_landmarks else None

def convert_landmarks_to_tensor(landmarks):
    """Convert MediaPipe pose landmarks to a tensor for the embedding model"""
    landmarks_list = [[lm.x, lm.y, lm.z] for lm in landmarks.landmark]
    return torch.tensor(landmarks_list).float().unsqueeze(0)

def extract_embeddings_from_dataset(dataset_path):
    """Extract pose embeddings from all images in the dataset"""
    embeddings = []
    image_paths = []
    pose_landmarks_list = []  # Store the actual landmarks for visualization

    print(f"Processing images from {dataset_path}...")
    
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path)
                
                if image is None:
                    print(f"Warning: Could not read image {image_path}")
                    continue
                    
                landmarks = extract_pose_landmarks(image)
                if landmarks:
                    # Convert landmarks to tensor for the model
                    landmarks_tensor = convert_landmarks_to_tensor(landmarks)
                    
                    # Get embedding from your trained model
                    with torch.no_grad():
                        embedding = model(landmarks_tensor)
                    
                    embeddings.append(embedding.squeeze().cpu().numpy())
                    image_paths.append(image_path)
                    pose_landmarks_list.append(landmarks)
                else:
                    print(f"No pose detected in {image_path}")

    return np.array(embeddings), image_paths, pose_landmarks_list

def visualize_pose_on_image(image_path, landmarks, output_path=None):
    """Draw the pose landmarks on an image"""
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create a copy for drawing
    annotated_image = image_rgb.copy()
    
    # Draw the pose landmarks
    mp_drawing.draw_landmarks(
        annotated_image, 
        landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1)
    )
    
    if output_path:
        # Convert back to BGR for saving with OpenCV
        cv2.imwrite(output_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    
    return annotated_image

def scale_embeddings(embeddings):
    """Scale the embeddings to have zero mean and unit variance"""
    scaler = StandardScaler()
    return scaler.fit_transform(embeddings)

def reduce_dimensions_with_pca(embeddings, n_components=50):
    """Reduce dimensionality while preserving variance"""
    # Make sure n_components doesn't exceed the size of the data
    n_components = min(n_components, embeddings.shape[0]-1, embeddings.shape[1])
    
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(embeddings)
    
    # Report how much variance is preserved
    explained_var = np.sum(pca.explained_variance_ratio_)
    print(f"PCA with {n_components} components explains {explained_var:.2%} of variance")
    
    return reduced

def find_optimal_eps(embeddings):
    """Find optimal eps value for DBSCAN using k-distance method"""
    # Calculate distances between all points
    distances = pdist(embeddings, metric='euclidean')
    distances = squareform(distances)
    
    # For each point, find distance to its k-th nearest neighbor
    k = min(5, len(embeddings)-1)
    k_distances = np.sort(distances, axis=1)[:, k]
    
    # Sort these k-distances
    sorted_k_distances = np.sort(k_distances)
    
    # Plot the k-distance graph
    plt.figure(figsize=(10, 5))
    plt.plot(sorted_k_distances)
    plt.axhline(y=np.median(sorted_k_distances), color='r', linestyle='--')
    plt.title(f"K-distance Graph (k={k})")
    plt.xlabel("Points sorted by distance to kth nearest neighbor")
    plt.ylabel("Distance")
    plt.grid(True)
    plt.savefig("k_distance_graph.png")
    plt.close()
    
    # Find the "elbow" point
    differences = np.diff(sorted_k_distances)
    elbow_index = np.argmax(differences) if len(differences) > 0 else len(sorted_k_distances) // 2
    eps_value = sorted_k_distances[elbow_index]
    
    print(f"Suggested eps value: {eps_value:.4f}")
    return eps_value

def cluster_pose_embeddings(embeddings, method="auto"):
    """Cluster the pose embeddings using either DBSCAN or K-means"""
    if method == "auto":
        # Try both methods and pick the best one
        return try_multiple_clustering_methods(embeddings)
    elif method == "dbscan":
        # Use DBSCAN with automatic eps selection
        eps = find_optimal_eps(embeddings)
        labels = DBSCAN(eps=eps, min_samples=3, metric='euclidean').fit(embeddings).labels_
        return labels
    elif method == "kmeans":
        # Use K-means with a fixed number of clusters
        n_clusters = estimate_number_of_clusters(embeddings)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(embeddings)
        return kmeans.labels_
    else:
        raise ValueError(f"Unknown clustering method: {method}")

def estimate_number_of_clusters(embeddings):
    """Estimate a good number of clusters for K-means"""
    # Try different numbers of clusters and evaluate with silhouette score
    max_clusters = min(20, len(embeddings) // 2)
    best_k = 2
    best_score = -1
    
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(embeddings)
        labels = kmeans.labels_
        
        try:
            score = silhouette_score(embeddings, labels)
            print(f"K-means with {k} clusters: silhouette score = {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_k = k
        except:
            pass
    
    print(f"Best number of clusters: {best_k} with silhouette score {best_score:.4f}")
    return best_k

def try_multiple_clustering_methods(embeddings):
    """Try different clustering methods and parameters"""
    results = []
    
    # Try DBSCAN with automatic eps
    eps = find_optimal_eps(embeddings)
    labels_dbscan = DBSCAN(eps=eps, min_samples=3, metric='euclidean').fit(embeddings).labels_
    num_clusters_dbscan = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
    
    # Calculate silhouette score if possible
    sil_dbscan = None
    if num_clusters_dbscan >= 2 and not all(l == -1 for l in labels_dbscan):
        non_noise = labels_dbscan != -1
        if sum(non_noise) > 1 and len(set(labels_dbscan[non_noise])) > 1:
            sil_dbscan = silhouette_score(embeddings[non_noise], labels_dbscan[non_noise])
            print(f"DBSCAN: {num_clusters_dbscan} clusters, silhouette={sil_dbscan:.4f}")
            results.append({
                "method": "DBSCAN",
                "labels": labels_dbscan,
                "score": sil_dbscan,
                "num_clusters": num_clusters_dbscan
            })
    
    # Try K-means with the estimated best number of clusters
    k = estimate_number_of_clusters(embeddings)
    labels_kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(embeddings).labels_
    
    try:
        sil_kmeans = silhouette_score(embeddings, labels_kmeans)
        print(f"K-means: {k} clusters, silhouette={sil_kmeans:.4f}")
        results.append({
            "method": "K-means",
            "labels": labels_kmeans,
            "score": sil_kmeans,
            "num_clusters": k
        })
    except:
        pass
    
    # Choose the best method
    if results:
        results.sort(key=lambda x: x["score"], reverse=True)
        best_result = results[0]
        print(f"Best clustering method: {best_result['method']} with {best_result['num_clusters']} clusters")
        return best_result["labels"]
    else:
        print("No successful clustering found. Defaulting to DBSCAN with eps=0.5")
        return DBSCAN(eps=0.5, min_samples=2).fit(embeddings).labels_

def visualize_tsne_with_poses(embeddings, labels, image_paths, pose_landmarks_list, output_dir="results"):
    """Visualize the t-SNE embeddings with pose examples from each cluster"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create t-SNE embedding
    perplexity = min(30, len(embeddings) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, learning_rate='auto', init='pca')
    reduced_embeddings = tsne.fit_transform(embeddings)
    
    # Get cluster information
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    
    # Create better color map
    cmap = plt.cm.get_cmap('tab20', max(20, len(unique_labels)))
    
    # Create t-SNE plot with clear cluster separation
    plt.figure(figsize=(12, 10))
    
    # Plot each cluster with better visual distinction
    for i, label in enumerate(unique_labels):
        mask = labels == label
        if label == -1:  # Noise points
            plt.scatter(
                reduced_embeddings[mask, 0], 
                reduced_embeddings[mask, 1],
                c='lightgray', s=30, alpha=0.5, label='Noise'
            )
        else:
            plt.scatter(
                reduced_embeddings[mask, 0], 
                reduced_embeddings[mask, 1],
                c=[cmap(i % 20)], s=80, alpha=0.8, 
                edgecolors='w', linewidths=0.8,
                label=f'Cluster {label}'
            )
    
    plt.title(f"t-SNE visualization of pose embeddings\n({n_clusters} clusters found)")
    plt.xlabel("t-SNE component 1")
    plt.ylabel("t-SNE component 2")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend if there aren't too many clusters
    if len(unique_labels) <= 10:
        plt.legend(loc='best', framealpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tsne_visualization.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a directory for pose examples
    poses_dir = os.path.join(output_dir, "pose_examples")
    os.makedirs(poses_dir, exist_ok=True)
    
    # Save pose examples from each cluster
    print("Saving pose examples from each cluster...")
    for label in unique_labels:
        if label == -1:
            continue  # Skip noise
            
        # Create a directory for this cluster
        cluster_dir = os.path.join(poses_dir, f"cluster_{label}")
        os.makedirs(cluster_dir, exist_ok=True)
        
        # Get indices of samples in this cluster
        indices = np.where(labels == label)[0]
        
        # Select a subset of samples to visualize
        n_samples = min(5, len(indices))
        sample_indices = np.random.choice(indices, size=n_samples, replace=False)
        
        # Save poses for these samples
        for i, idx in enumerate(sample_indices):
            image_path = image_paths[idx]
            landmarks = pose_landmarks_list[idx]
            
            # Create an image with the pose overlaid
            output_path = os.path.join(cluster_dir, f"sample_{i+1}.jpg")
            visualize_pose_on_image(image_path, landmarks, output_path)
    
    # Create a grid visualization of poses by cluster
    visualize_pose_grid(labels, image_paths, pose_landmarks_list, output_dir)
    
    # Create a report
    with open(os.path.join(output_dir, "clustering_report.txt"), "w") as f:
        f.write(f"Number of clusters: {n_clusters}\n")
        f.write(f"Total samples: {len(embeddings)}\n\n")
        
        f.write("Cluster statistics:\n")
        for label in unique_labels:
            if label == -1:
                f.write(f"  Noise: {np.sum(labels == label)} samples\n")
            else:
                f.write(f"  Cluster {label}: {np.sum(labels == label)} samples\n")

def visualize_pose_grid(labels, image_paths, pose_landmarks_list, output_dir):
    """Create a grid visualization of poses by cluster"""
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    unique_labels = [l for l in unique_labels if l != -1]  # Remove noise
    
    if not unique_labels:
        print("No valid clusters to visualize")
        return
    
    # Determine grid layout
    n_clusters = len(unique_labels)
    samples_per_cluster = 3
    
    # Create the figure
    fig, axes = plt.subplots(n_clusters, samples_per_cluster, 
                            figsize=(samples_per_cluster*3, n_clusters*3))
    
    # Handle the case with just one cluster
    if n_clusters == 1:
        axes = np.array([axes])
    
    # Fill the grid
    for i, label in enumerate(unique_labels):
        indices = np.where(labels == label)[0]
        n_samples = min(samples_per_cluster, len(indices))
        
        # Get random samples from this cluster
        sample_indices = np.random.choice(indices, size=n_samples, replace=False)
        
        for j, idx in enumerate(sample_indices):
            # Get the image and landmarks
            image_path = image_paths[idx]
            landmarks = pose_landmarks_list[idx]
            
            # Create pose visualization
            pose_img = visualize_pose_on_image(image_path, landmarks)
            
            # Display in the grid
            if n_clusters == 1:
                ax = axes[j]
            else:
                ax = axes[i, j]
                
            ax.imshow(pose_img)
            ax.set_title(f"Cluster {label}")
            ax.axis('off')
        
        # Turn off empty axes
        for j in range(n_samples, samples_per_cluster):
            if n_clusters == 1:
                axes[j].axis('off')
            else:
                axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pose_grid.png"), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Get dataset path
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_ETHZ", "seq2")
    output_dir = "pose_clustering_results"
    
    print(f"Analyzing pose data from {dataset_path}...")
    print(f"Results will be saved to {output_dir}")
    
    # Extract pose embeddings
    embeddings, image_paths, pose_landmarks_list = extract_embeddings_from_dataset(dataset_path)
    print(f"Extracted {len(embeddings)} pose embeddings")
    
    if len(embeddings) == 0:
        print("No valid pose embeddings found. Check your dataset path and pose detection parameters.")
        exit(1)
    
    # Analyze embedding distribution
    print("Analyzing embedding distribution...")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Mean embedding magnitude: {np.mean([np.linalg.norm(e) for e in embeddings]):.4f}")
    
    # Scale the embeddings
    embeddings_scaled = scale_embeddings(embeddings)
    
    # Reduce dimensions with PCA
    embeddings_pca = reduce_dimensions_with_pca(embeddings_scaled, n_components=min(50, len(embeddings)-1))
    
    # Cluster the embeddings
    print("Clustering pose embeddings...")
    labels = cluster_pose_embeddings(embeddings_pca, method="auto")
    
    # Visualize the results
    print("Creating visualizations...")
    visualize_tsne_with_poses(embeddings_pca, labels, image_paths, pose_landmarks_list, output_dir)
    
    print(f"Clustering complete! Results saved to {output_dir}")