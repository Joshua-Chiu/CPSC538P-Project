import os
import torch
import numpy as np
import cv2
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from train_triplets import PoseEmbeddingNet
import mediapipe as mp
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

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

def visualize_distance_distribution(embeddings, output_dir):
    """Visualize the distribution of distances between embeddings to help with clustering"""
    # Calculate pairwise distances
    print("Calculating pairwise distances...")
    distances = pdist(embeddings, metric='euclidean')
    
    # Plot distance distribution
    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=50, alpha=0.7)
    plt.axvline(x=np.median(distances), color='r', linestyle='--', label=f'Median: {np.median(distances):.2f}')
    plt.axvline(x=np.percentile(distances, 10), color='g', linestyle='--', label=f'10th percentile: {np.percentile(distances, 10):.2f}')
    plt.axvline(x=np.percentile(distances, 90), color='b', linestyle='--', label=f'90th percentile: {np.percentile(distances, 90):.2f}')
    plt.title('Distribution of Pairwise Distances Between Pose Embeddings')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "distance_distribution.png"), dpi=300)
    plt.close()
    
    # Plot k-distance graph for DBSCAN
    k = min(20, len(embeddings) - 1)  # Choose a reasonable k value
    neigh = NearestNeighbors(n_neighbors=k+1).fit(embeddings)  # +1 because the first neighbor is the point itself
    distances, _ = neigh.kneighbors(embeddings)
    
    # Sort the distances to the k-th neighbor
    k_distances = np.sort(distances[:, k])
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_distances)
    plt.title(f'K-distance graph (k={k})')
    plt.xlabel('Points sorted by distance to k-th nearest neighbor')
    plt.ylabel(f'Distance to {k}-th nearest neighbor')
    plt.grid(True, alpha=0.3)
    
    # Find potential "elbow" points
    from scipy.signal import argrelextrema
    k_distances_smoothed = np.convolve(k_distances, np.ones(5)/5, mode='valid')  # Smooth the curve
    local_maxima = argrelextrema(np.gradient(k_distances_smoothed), np.greater)[0]
    
    if len(local_maxima) > 0:
        for i, idx in enumerate(local_maxima):
            if idx < len(k_distances):
                plt.axvline(x=idx, color='r', linestyle='--', alpha=0.5)
                plt.text(idx, k_distances[idx], f'Potential eps: {k_distances[idx]:.2f}', 
                        rotation=90, verticalalignment='bottom')
    
    plt.savefig(os.path.join(output_dir, "k_distance_graph.png"), dpi=300)
    plt.close()
    
    # Create a hierarchical clustering dendrogram to visualize the structure
    if len(embeddings) <= 1000:  # Only for reasonably sized datasets
        print("Creating hierarchical clustering dendrogram...")
        Z = linkage(embeddings, method='ward')
        
        plt.figure(figsize=(12, 8))
        dendrogram(Z, truncate_mode='lastp', p=30, leaf_rotation=90.)
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('Cluster size')
        plt.ylabel('Distance')
        plt.axhline(y=np.median(Z[:, 2]), color='r', linestyle='--', 
                   label=f'Median distance: {np.median(Z[:, 2]):.2f}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(output_dir, "dendrogram.png"), dpi=300)
        plt.close()
    
    return {
        'median_distance': np.median(distances),
        'distance_10th': np.percentile(distances, 10),
        'distance_90th': np.percentile(distances, 90),
        'k_distances': k_distances
    }

def find_optimal_number_of_clusters(embeddings):
    """Use various methods to estimate the optimal number of clusters"""
    print("Estimating optimal number of clusters...")
    
    # Store results from different methods
    results = {}
    
    # 1. Silhouette analysis for KMeans clustering
    sil_scores = []
    ch_scores = []
    range_n_clusters = range(2, min(50, len(embeddings) // 2))
    
    for n_clusters in range_n_clusters:
        # Initialize the clusterer
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = clusterer.fit_predict(embeddings)
        
        # Silhouette score
        try:
            sil_score = silhouette_score(embeddings, cluster_labels)
            sil_scores.append(sil_score)
            print(f"  K-means with {n_clusters} clusters: silhouette={sil_score:.4f}")
        except:
            sil_scores.append(-1)
        
        # Calinski-Harabasz score
        try:
            ch_score = calinski_harabasz_score(embeddings, cluster_labels)
            ch_scores.append(ch_score)
        except:
            ch_scores.append(-1)
    
    results['kmeans_silhouette'] = sil_scores
    results['kmeans_ch'] = ch_scores
    
    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(list(range_n_clusters), sil_scores, 'o-', label='Silhouette Score')
    
    # Find the best n_clusters
    if sil_scores:
        best_n_clusters_sil = range_n_clusters[np.argmax(sil_scores)]
        plt.axvline(x=best_n_clusters_sil, color='r', linestyle='--', 
                   label=f'Best k: {best_n_clusters_sil}')
        
        results['best_n_clusters_silhouette'] = best_n_clusters_sil
        print(f"  Best k according to silhouette score: {best_n_clusters_sil}")
    
    plt.title('Silhouette Score vs Number of Clusters')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join("pose_clustering_results", "silhouette_scores.png"), dpi=300)
    plt.close()
    
    # Plot Calinski-Harabasz scores
    plt.figure(figsize=(10, 6))
    plt.plot(list(range_n_clusters), ch_scores, 'o-', label='Calinski-Harabasz Score')
    
    # Find the best n_clusters
    if ch_scores:
        best_n_clusters_ch = range_n_clusters[np.argmax(ch_scores)]
        plt.axvline(x=best_n_clusters_ch, color='r', linestyle='--', 
                   label=f'Best k: {best_n_clusters_ch}')
        
        results['best_n_clusters_ch'] = best_n_clusters_ch
        print(f"  Best k according to Calinski-Harabasz score: {best_n_clusters_ch}")
    
    plt.title('Calinski-Harabasz Score vs Number of Clusters')
    plt.xlabel('Number of clusters')
    plt.ylabel('Calinski-Harabasz Score')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join("pose_clustering_results", "ch_scores.png"), dpi=300)
    plt.close()
    
    # 2. Hierarchical clustering analysis
    Z = linkage(embeddings, method='ward')
    last_merge_distances = Z[:, 2]
    
    # Look for the largest jump in merge distances - this can indicate natural clusters
    merge_diff = np.diff(last_merge_distances)
    
    if len(merge_diff) > 0:
        max_jump_idx = np.argmax(merge_diff)
        suggested_n_clusters = len(embeddings) - max_jump_idx
        
        # Bound the suggestion to a reasonable range
        suggested_n_clusters = min(max(suggested_n_clusters, 2), 50)
        
        results['hierarchical_suggested'] = suggested_n_clusters
        print(f"  Hierarchical clustering suggests {suggested_n_clusters} clusters")
    
    # 3. DBSCAN-based estimate
    # Get k-distances
    k = min(20, len(embeddings) - 1)
    neigh = NearestNeighbors(n_neighbors=k+1).fit(embeddings)
    distances, _ = neigh.kneighbors(embeddings)
    
    # Try to find natural breakpoints in the k-distance graph
    k_dist = np.sort(distances[:, k])
    k_dist_diff = np.gradient(k_dist)
    
    # Find points with large gradients - these are potential eps values
    threshold = np.percentile(k_dist_diff, 90)  # Get the top 10% of gradients
    potential_eps_indices = np.where(k_dist_diff > threshold)[0]
    
    dbscan_clusters = []
    
    # Try a few eps values from the k-distance graph
    for idx in potential_eps_indices:
        if idx < len(k_dist):
            eps = k_dist[idx]
            dbscan = DBSCAN(eps=eps, min_samples=3).fit(embeddings)
            labels = dbscan.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_ratio = np.sum(labels == -1) / len(labels)
            
            if n_clusters >= 2 and noise_ratio < 0.5:  # Skip if too much noise or too few clusters
                dbscan_clusters.append((eps, n_clusters, noise_ratio))
                print(f"  DBSCAN with eps={eps:.4f}: {n_clusters} clusters, {noise_ratio:.1%} noise")
    
    if dbscan_clusters:
        # Sort by noise ratio (ascending) and number of clusters (closest to silhouette best)
        target_clusters = results.get('best_n_clusters_silhouette', 10)
        dbscan_clusters.sort(key=lambda x: (x[2], abs(x[1] - target_clusters)))
        best_eps, best_n_clusters, _ = dbscan_clusters[0]
        
        results['dbscan_suggested'] = best_n_clusters
        results['dbscan_eps'] = best_eps
        print(f"  DBSCAN suggests {best_n_clusters} clusters with eps={best_eps:.4f}")
    
    # Combine results and suggest a final number
    suggested_n_clusters = []
    
    if 'best_n_clusters_silhouette' in results:
        suggested_n_clusters.append(results['best_n_clusters_silhouette'])
    
    if 'best_n_clusters_ch' in results:
        suggested_n_clusters.append(results['best_n_clusters_ch'])
    
    if 'hierarchical_suggested' in results:
        suggested_n_clusters.append(results['hierarchical_suggested'])
    
    if 'dbscan_suggested' in results:
        suggested_n_clusters.append(results['dbscan_suggested'])
    
    if suggested_n_clusters:
        # Take the median for a consensus
        final_n_clusters = int(np.median(suggested_n_clusters))
        results['final_suggestion'] = final_n_clusters
        print(f"Final suggested number of clusters: {final_n_clusters}")
    else:
        # Fallback
        results['final_suggestion'] = 10
        print("Could not determine optimal clusters, using default value of 10")
    
    return results

def adaptive_clustering(embeddings, distance_stats):
    """Adaptively cluster embeddings using multiple methods and determine the best one"""
    print("Performing adaptive clustering...")
    
    # 1. First, estimate the number of clusters in the data
    cluster_analysis = find_optimal_number_of_clusters(embeddings)
    suggested_n_clusters = cluster_analysis.get('final_suggestion', 10)
    
    methods_to_try = []
    
    # 2. Try K-means with the suggested number of clusters
    methods_to_try.append({
        'name': f'KMeans-{suggested_n_clusters}',
        'labels': KMeans(n_clusters=suggested_n_clusters, random_state=42, n_init=10).fit_predict(embeddings)
    })
    
    # 3. Try Agglomerative clustering with the suggested number of clusters
    methods_to_try.append({
        'name': f'Agglomerative-{suggested_n_clusters}',
        'labels': AgglomerativeClustering(n_clusters=suggested_n_clusters, linkage='ward').fit_predict(embeddings)
    })
    
    # 4. Try DBSCAN with automatic eps from the k-distance graph
    if 'dbscan_eps' in cluster_analysis:
        eps = cluster_analysis['dbscan_eps']
        labels_dbscan = DBSCAN(eps=eps, min_samples=3).fit_predict(embeddings)
        methods_to_try.append({
            'name': f'DBSCAN-eps{eps:.2f}',
            'labels': labels_dbscan
        })
    
    # 5. Try DBSCAN with a range of eps values
    # Use the distance stats to get reasonable eps values
    eps_values = [
        distance_stats['distance_10th'] * 0.5,  # Very tight clusters
        distance_stats['distance_10th'],        # Tight clusters
        distance_stats['distance_10th'] * 1.5,  # Moderate clusters
        distance_stats['distance_10th'] * 2.0   # Looser clusters
    ]
    
    for eps in eps_values:
        labels_dbscan = DBSCAN(eps=eps, min_samples=3).fit_predict(embeddings)
        n_clusters = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
        noise_ratio = np.sum(labels_dbscan == -1) / len(labels_dbscan)
        
        if n_clusters >= 2 and noise_ratio < 0.5:  # Skip if too much noise or too few clusters
            methods_to_try.append({
                'name': f'DBSCAN-eps{eps:.2f}',
                'labels': labels_dbscan
            })
    
    # 6. Evaluate each clustering method
    results = []
    
    for method in methods_to_try:
        labels = method['labels']
        name = method['name']
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        # Skip methods that produced only one cluster
        if n_clusters < 2:
            print(f"  {name}: Only {n_clusters} clusters found, skipping evaluation")
            continue
        
        # Skip methods with too many noise points (for DBSCAN)
        if name.startswith('DBSCAN'):
            noise_ratio = np.sum(labels == -1) / len(labels)
            if noise_ratio > 0.4:  # Skip if more than 40% noise
                print(f"  {name}: {noise_ratio:.1%} noise points, skipping evaluation")
                continue
        
        # For DBSCAN, calculate scores only on non-noise points
        if name.startswith('DBSCAN'):
            non_noise = labels != -1
            if np.sum(non_noise) > 1 and len(set(labels[non_noise])) > 1:
                try:
                    sil_score = silhouette_score(embeddings[non_noise], labels[non_noise])
                    ch_score = calinski_harabasz_score(embeddings[non_noise], labels[non_noise])
                except:
                    sil_score = -1
                    ch_score = -1
            else:
                sil_score = -1
                ch_score = -1
        else:
            try:
                sil_score = silhouette_score(embeddings, labels)
                ch_score = calinski_harabasz_score(embeddings, labels)
            except:
                sil_score = -1
                ch_score = -1
        
        # Calculate a combined score
        if sil_score > 0 and ch_score > 0:
            # Normalize CH score (it can get very large)
            normalized_ch = ch_score / (ch_score + 1000)  # Soft normalization
            combined_score = (sil_score + normalized_ch) / 2
        else:
            combined_score = max(sil_score, 0)  # Use silhouette only if both aren't available
        
        # Bias slightly towards methods that found more clusters (closer to suggested_n_clusters)
        # But don't let this dominate the actual quality scores
        cluster_count_bonus = 1.0 - min(abs(n_clusters - suggested_n_clusters) / suggested_n_clusters, 0.2)
        final_score = combined_score * cluster_count_bonus
        
        results.append({
            'name': name,
            'labels': labels,
            'n_clusters': n_clusters,
            'silhouette': sil_score,
            'ch_score': ch_score,
            'combined_score': combined_score,
            'final_score': final_score
        })
        
        print(f"  {name}: {n_clusters} clusters, silhouette={sil_score:.4f}, CH={ch_score:.2f}, score={final_score:.4f}")
    
    # 7. Choose the best method
    if results:
        results.sort(key=lambda x: x['final_score'], reverse=True)
        best_method = results[0]
        print(f"Best clustering method: {best_method['name']} with {best_method['n_clusters']} clusters")
        print(f"  Silhouette score: {best_method['silhouette']:.4f}")
        print(f"  CH score: {best_method['ch_score']:.2f}")
        
        return best_method['labels'], results
    else:
        print("No successful clustering found. Falling back to KMeans with 10 clusters.")
        return KMeans(n_clusters=10, random_state=42, n_init=10).fit_predict(embeddings), []

def visualize_tsne_embeddings(embeddings, labels, image_paths, pose_landmarks_list, output_dir="pose_clustering_results"):
    """Visualize the t-SNE embeddings with pose examples from each cluster"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create t-SNE embedding with improved parameters for better separation
    print("Creating t-SNE visualization with improved parameters...")
    
    # Try different perplexity values
    perplexities = [5, 15, 30, 50]
    best_separation = -float('inf')
    best_embedding = None
    best_perplexity = None
    
    for perp in perplexities:
        if perp >= len(embeddings):
            continue
            
        try:
            # Create t-SNE embedding
            tsne = TSNE(
                n_components=2, 
                perplexity=perp, 
                random_state=42, 
                learning_rate='auto',
                n_iter=3000,
                init='pca'
            )
            reduced = tsne.fit_transform(embeddings)
            
            # Calculate cluster separation in t-SNE space
            unique_labels = np.unique(labels)
            if len(unique_labels) <= 1:
                continue
                
            # Calculate centroids for each cluster
            centroids = np.array([
                reduced[labels == label].mean(axis=0)
                for label in unique_labels if label != -1 and np.sum(labels == label) > 0
            ])
            
            if len(centroids) <= 1:
                continue
            
            # Calculate average distance between centroids
            centroid_distances = pdist(centroids)
            if len(centroid_distances) == 0:
                continue
                
            inter_cluster_dist = np.mean(centroid_distances)
            
            # Calculate average intra-cluster spread
            intra_cluster_dists = []
            for label in unique_labels:
                if label == -1 or np.sum(labels == label) <= 1:
                    continue
                points = reduced[labels == label]
                centroid = points.mean(axis=0)
                distances = np.sqrt(np.sum((points - centroid)**2, axis=1))
                intra_cluster_dists.append(np.mean(distances))
            
            if not intra_cluster_dists:
                continue
                
            intra_cluster_dist = np.mean(intra_cluster_dists)
            
            # Separation metric: inter-cluster distance / intra-cluster distance
            if intra_cluster_dist > 0:
                separation = inter_cluster_dist / intra_cluster_dist
                print(f"  t-SNE with perplexity={perp}: separation={separation:.2f}")
                
                if separation > best_separation:
                    best_separation = separation
                    best_embedding = reduced
                    best_perplexity = perp
        except Exception as e:
            print(f"  Error with perplexity={perp}: {e}")
    
    if best_embedding is None:
        print("Using default t-SNE parameters")
        perp = min(30, len(embeddings)-1)
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
        best_embedding = tsne.fit_transform(embeddings)
        best_perplexity = perp
    
    reduced_embeddings = best_embedding
    print(f"Using t-SNE with perplexity={best_perplexity}")
    
    # Get cluster information
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    
    # Create a colormap
    from matplotlib.colors import ListedColormap
    
    # Use a high-contrast colormap for better distinction
    base_colors = plt.cm.tab20(np.linspace(0, 1, 20))
    if len(unique_labels) > 20:
        # Add more colors if needed
        extra_colors = plt.cm.tab20b(np.linspace(0, 1, 20))
        all_colors = np.vstack([base_colors, extra_colors])
        cmap = ListedColormap(all_colors)
    else:
        cmap = plt.cm.tab20
    
    # Create t-SNE plot with clear cluster separation
    plt.figure(figsize=(14, 12))
    
    # Plot each cluster
    for i, label in enumerate(unique_labels):
        mask = labels == label
        if label == -1:  # Noise points
            plt.scatter(
                reduced_embeddings[mask, 0], 
                reduced_embeddings[mask, 1],
                c='lightgray', s=20, alpha=0.5, marker='x', label='Noise'
            )
        else:
            color_idx = i % cmap.N
            plt.scatter(
                reduced_embeddings[mask, 0], 
                reduced_embeddings[mask, 1],
                c=[cmap(color_idx)], s=70, alpha=0.8, 
                edgecolors='w', linewidths=0.8,
                label=f'Cluster {label}'
            )
    
    plt.title(f"t-SNE visualization of pose embeddings\n({n_clusters} clusters found)")
    plt.xlabel("t-SNE component 1")
    plt.ylabel("t-SNE component 2")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Add legend but keep it manageable
    if len(unique_labels) <= 15:
        plt.legend(loc='best', framealpha=0.7)
    else:
        # With many clusters, don't show individual legend entries
        handles, labels = plt.gca().get_legend_handles_labels()
        if -1 in unique_labels:  # Add just the noise label if present
            noise_idx = list(labels).index('Noise')
            plt.legend([handles[noise_idx]], [labels[noise_idx]], loc='best', framealpha=0.7)
        else:
            plt.legend([], [], title=f"{n_clusters} clusters", loc='best', framealpha=0.7)
    
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
    
    return labels

if __name__ == "__main__":
    # Create output directory
    output_dir = "pose_clustering_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get dataset path
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_ETHZ", "seq2")
    
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
    
    # Scale the embeddings - important for distance-based methods
    print("Scaling embeddings...")
    embeddings_scaled = scale_embeddings(embeddings)
    
    # Analyze the distribution of distances between embeddings
    distance_stats = visualize_distance_distribution(embeddings_scaled, output_dir)
    
    # Perform adaptive clustering to find the natural number of clusters
    labels, clustering_results = adaptive_clustering(embeddings_scaled, distance_stats)
    
    # Count actual number of clusters
    unique_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Found {unique_clusters} clusters")
    
    # Visualize the results with t-SNE
    print("Creating visualizations...")
    visualize_tsne_embeddings(embeddings_scaled, labels, image_paths, pose_landmarks_list, output_dir)
    
    print(f"Clustering complete! Results saved to {output_dir}")
    print(f"Check {output_dir}/tsne_visualization.png to see the clustering results")

def visualize_pose_grid(labels, image_paths, pose_landmarks_list, output_dir):
    """Create a grid visualization of poses by cluster"""
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    unique_labels = [l for l in unique_labels if l != -1]  # Remove noise
    
    if not unique_labels:
        print("No valid clusters to visualize")
        return
    
    # Determine grid layout - show up to 20 clusters with 3 samples each
    n_clusters = min(20, len(unique_labels))
    samples_per_cluster = 3
    
    # Create the figure
    fig, axes = plt.subplots(n_clusters, samples_per_cluster, 
                            figsize=(samples_per_cluster*3, n_clusters*3))
    
    # Handle the case with just one cluster
    if n_clusters == 1:
        axes = np.array([axes])
    
    # Fill the grid
    for i, label in enumerate(unique_labels[:n_clusters]):  # Only show first 20 clusters
        indices = np.where(labels == label)[0]
        n_samples = min(samples_per_cluster, len(indices))
        
        # Get random samples from this cluster
        if len(indices) <= samples_per_cluster:
            sample_indices = indices
        else:
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