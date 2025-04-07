import torch
import ast
import os
import cv2
import mediapipe as mp
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
from tqdm import tqdm  # Progress bar
from concurrent.futures import ProcessPoolExecutor  # For multiprocessing
from train_triplets import PoseEmbeddingNet  # Import your model definition file

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load the pre-trained pose embedding model
pose_embedding_model = PoseEmbeddingNet(input_size=99, embedding_size=128)
pose_embedding_model.load_state_dict(torch.load('pose_embedding_model_fine_tuned.pth', weights_only=True))
pose_embedding_model.eval()

def extract_pose_landmarks(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        landmarks = [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
        return torch.tensor(landmarks).flatten()
    else:
        return None

def batch_extract_landmarks(image_paths):
    """Function to process landmarks in parallel using multiprocessing."""
    with ProcessPoolExecutor() as executor:
        return list(executor.map(extract_pose_landmarks, image_paths))

def evaluate_model(image_pairs, dataset_path):
    dataset_name = os.path.basename(dataset_path.rstrip("/\\"))
    true_labels = []
    predicted_scores = []
    skipped_files = 0
    total_files = len(image_pairs)
    correct_positive = 0
    correct_negative = 0
    all_embeddings = []
    all_labels = []

    # Get all unique image paths for landmark extraction
    all_img_paths = list(set([p for pair in image_pairs for p in pair[:2]]))
    
    # Extract landmarks in parallel
    pose_dict = dict(zip(all_img_paths, batch_extract_landmarks(all_img_paths)))

    for img1_path, img2_path, label in tqdm(image_pairs, desc="Evaluating Pairs", unit="pair"):
        landmarks1 = pose_dict.get(img1_path)
        landmarks2 = pose_dict.get(img2_path)

        if landmarks1 is None or landmarks2 is None:
            skipped_files += 1
            continue

        embedding1 = pose_embedding_model(landmarks1.unsqueeze(0))
        embedding2 = pose_embedding_model(landmarks2.unsqueeze(0))

        all_embeddings.append(embedding1.detach().numpy())
        all_embeddings.append(embedding2.detach().numpy())
        all_labels.append(label)
        all_labels.append(label)

        similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2)
        true_labels.append(label)
        predicted_scores.append(similarity.item())

        if (similarity.item() > 0.5 and label == 1) or (similarity.item() <= 0.5 and label == 0):
            if label == 1:
                correct_positive += 1
            else:
                correct_negative += 1

    print(f"Skipped {skipped_files} files out of {total_files} due to no landmarks detected.")

    fpr, tpr, thresholds = roc_curve(true_labels, predicted_scores)
    auc_score = auc(fpr, tpr)

    print(f"Correctly identified {correct_positive} positive pairs.")
    print(f"Correctly identified {correct_negative} negative pairs.")
    print(f"Total true positive pairs: {true_labels.count(1)}")
    print(f"Total true negative pairs: {true_labels.count(0)}")

    all_embeddings = np.vstack(all_embeddings)
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(all_embeddings)

    # Save t-SNE plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=all_labels, cmap='coolwarm', alpha=0.7)
    plt.colorbar(scatter, label='True Label')
    plt.title('t-SNE Visualization of Pose Embeddings')
    plt.xlabel('t-SNE component 1')
    plt.ylabel('t-SNE component 2')
    tsne_filename = f"{dataset_name} tsne.png"
    plt.savefig(tsne_filename)
    print(f"t-SNE plot saved as {tsne_filename}")
    plt.close()

    # Nearest Neighbor Accuracy
    dist_matrix = cdist(all_embeddings, all_embeddings, metric='cosine')
    np.fill_diagonal(dist_matrix, np.inf)
    nearest_neighbors = np.argmin(dist_matrix, axis=1)

    correct_nn = sum(1 for i in range(len(all_labels)) if all_labels[i] == all_labels[nearest_neighbors[i]])
    nn_accuracy = correct_nn / len(all_labels)
    print(f"Nearest Neighbor Accuracy: {nn_accuracy * 100:.2f}%")

    # Save ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    roc_filename = f"{dataset_name} roc.png"
    plt.savefig(roc_filename)
    print(f"ROC curve saved as {roc_filename}")
    plt.close()

    # Save summary to text file
    summary_filename = f"{dataset_name} evaluation.txt"
    with open(summary_filename, "w") as f:
        f.write(f"Skipped files: {skipped_files}/{total_files}\n")
        f.write(f"Correctly identified positive pairs: {correct_positive}\n")
        f.write(f"Correctly identified negative pairs: {correct_negative}\n")
        f.write(f"Total true positives: {true_labels.count(1)}\n")
        f.write(f"Total true negatives: {true_labels.count(0)}\n")
        f.write(f"AUC Score: {auc_score:.4f}\n")
        f.write(f"Nearest Neighbor Accuracy: {nn_accuracy * 100:.2f}%\n")

    print(f"Evaluation summary saved as {summary_filename}")

    return fpr, tpr, auc_score

def load_image_pairs(filepath):
    pairs = []
    with open(filepath, 'r') as f:
        for line in f:
            try:
                img1, img2, label = ast.literal_eval(line.strip())
                pairs.append((img1.strip(), img2.strip(), int(label)))
            except (ValueError, SyntaxError):
                print(f"⚠️ Skipping malformed line: {line.strip()}")
    return pairs

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, "dataset_ETHZ", "seq3")
    pairs_path = os.path.join(current_dir, "evaluation_pairs.txt")

    image_pairs = load_image_pairs(pairs_path)
    if not image_pairs:
        print("❌ No image pairs loaded. Please check the file path and contents.")
        exit()

    fpr, tpr, auc_score = evaluate_model(image_pairs, dataset_path)
    print(f"AUC Score: {auc_score}")
