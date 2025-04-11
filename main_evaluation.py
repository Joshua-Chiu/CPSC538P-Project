import torch
import ast
import os
import cv2
import mediapipe as mp
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score, mean_squared_error, precision_score, recall_score, matthews_corrcoef, precision_recall_curve, average_precision_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from train_triplets import PoseEmbeddingNet
import time

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9)

# Load the pre-trained pose embedding model
pose_embedding_model = PoseEmbeddingNet(input_size=99, embedding_size=128)
pose_embedding_model.load_state_dict(torch.load('pose_embedding_model.pth', weights_only=True))
pose_embedding_model.eval()

def extract_pose_landmarks(image_path):
    try:
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
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def batch_extract_landmarks(image_paths, use_multiprocessing=True):
    # If use_multiprocessing is True, use ProcessPoolExecutor; else use ThreadPoolExecutor
    executor_class = ProcessPoolExecutor if use_multiprocessing else ThreadPoolExecutor
    with executor_class(max_workers=4) as executor:
        return list(executor.map(extract_pose_landmarks, image_paths))

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

def evaluate_model(image_pairs, dataset_path, batch_size=5000):
    start_time = time.time()
    dataset_name = os.path.basename(dataset_path.rstrip("/\\"))
    true_labels = []
    predicted_scores = []
    nearest_neighbor_scores = []  # Add list for nearest neighbor scores
    skipped_files = 0
    total_files = len(image_pairs)
    correct_positive = 0
    correct_negative = 0
    total_true_positives = 0
    total_true_negatives = 0
    all_embeddings = []
    all_labels = []

    # Extract landmarks for all images in pairs
    all_img_paths = list(set([p for pair in image_pairs for p in pair[:2]]))
    pose_dict = dict(zip(all_img_paths, batch_extract_landmarks(all_img_paths, use_multiprocessing=True)))

    for idx, (img1_path, img2_path, label) in enumerate(tqdm(image_pairs, desc="Evaluating Pairs", unit="pair")):
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

        # Cosine similarity
        similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2)
        true_labels.append(label)
        predicted_scores.append(similarity.item())

        # Nearest Neighbor Score (Euclidean distance between embeddings)
        nn_score = pairwise_distances(embedding1.detach().numpy(), embedding2.detach().numpy(), metric='cosine')[0][0]
        nearest_neighbor_scores.append(nn_score)  # Store nearest neighbor score

        # Calculate true positive/negative based on predicted similarity
        if label == 1:  # True positive
            total_true_positives += 1
        if label == 0:  # True negative
            total_true_negatives += 1

        if similarity.item() > 0.5:  # Predicted same person
            if label == 1:
                correct_positive += 1
        else:  # Predicted different person
            if label == 0:
                correct_negative += 1

    total_time = time.time() - start_time
    print(f"✅ Finished processing all image pairs. Total time: {total_time/60:.2f} minutes.")

    if not all_embeddings:
        print("❌ No valid embeddings were generated. Exiting early.")
        return None, None, None, None

    # t-SNE visualization
    print("🚦 Starting t-SNE visualization...")
    all_embeddings = np.vstack(all_embeddings)

    sample_size = min(10000, len(all_embeddings))
    indices = np.random.choice(len(all_embeddings), size=sample_size, replace=False)
    sampled_embeddings = all_embeddings[indices]
    sampled_labels = [all_labels[i] for i in indices]

    tsne_sample = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_results = tsne_sample.fit_transform(sampled_embeddings)

    # Save t-SNE plot
    plt.figure(figsize=(10, 8))
    tsne_results = np.array(tsne_results)
    sampled_labels = np.array(sampled_labels)
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=sampled_labels, cmap='coolwarm', alpha=0.6)
    plt.colorbar(scatter)
    plt.title("t-SNE Visualization of Pose Embeddings")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.tight_layout()
    plt.savefig(f"{dataset_name}_tsne.png")
    plt.close()

    # ROC Curve
    true_labels = np.array(true_labels)
    predicted_scores = np.array(predicted_scores)

    bins = np.linspace(0, 1, num=200)
    digitized_scores = np.digitize(predicted_scores, bins) / len(bins)

    fpr, tpr, thresholds = roc_curve(true_labels, digitized_scores)
    auc_score = auc(fpr, tpr)

    # Confusion Matrix and F1 Score
    cm = confusion_matrix(true_labels, (predicted_scores > 0.5).astype(int))  # Using 0.5 as decision threshold
    f1 = f1_score(true_labels, (predicted_scores > 0.5).astype(int))

    # Precision and Recall
    precision = precision_score(true_labels, (predicted_scores > 0.5).astype(int))
    recall = recall_score(true_labels, (predicted_scores > 0.5).astype(int))

    # Matthews Correlation Coefficient (MCC)
    mcc = matthews_corrcoef(true_labels, (predicted_scores > 0.5).astype(int))

    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mean_squared_error(true_labels, predicted_scores))

    # Precision-Recall Curve
    precision_vals, recall_vals, _ = precision_recall_curve(true_labels, predicted_scores)
    average_precision = average_precision_score(true_labels, predicted_scores)

    print(f"Correctly identified {correct_positive} positive pairs.")
    print(f"Correctly identified {correct_negative} negative pairs.")
    print(f"Total true positives (ground truth positives): {total_true_positives}")
    print(f"Total true negatives (ground truth negatives): {total_true_negatives}")
    print(f"AUC Score: {auc_score:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"MCC: {mcc:.4f}")
    print(f"Average Precision (PR Curve AUC): {average_precision:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Confusion Matrix: \n{cm}")

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.savefig(f"{dataset_name}_roc.png")
    plt.close()

    # Precision-Recall Curve
    plt.figure()
    plt.plot(recall_vals, precision_vals, color='b', lw=2, label=f'PR curve (AP = {average_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.savefig(f"{dataset_name}_pr_curve.png")
    plt.close()

    # Nearest Neighbor Score
    avg_nn_score = np.mean(nearest_neighbor_scores)
    print(f"Average Nearest Neighbor Score (Cosine Distance): {avg_nn_score:.4f}")

    summary_filename = f"{dataset_name} evaluation.txt"
    with open(summary_filename, "w") as f:
        f.write(f"Skipped files: {skipped_files}/{total_files}\n")
        f.write(f"Correctly identified positive pairs: {correct_positive}\n")
        f.write(f"Correctly identified negative pairs: {correct_negative}\n")
        f.write(f"Total true positives (ground truth positives): {total_true_positives}\n")
        f.write(f"Total true negatives (ground truth negatives): {total_true_negatives}\n")
        f.write(f"AUC Score: {auc_score:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"MCC: {mcc:.4f}\n")
        f.write(f"Average Precision (PR Curve AUC): {average_precision:.4f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}\n")
        f.write(f"Confusion Matrix:\n{cm}\n")
        f.write(f"Average Nearest Neighbor Score (Cosine Distance): {avg_nn_score:.4f}\n")

    print(f"📄 Evaluation summary saved as {summary_filename}")

    return fpr, tpr, auc_score, avg_nn_score


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, "entireid", "bounding_box_test")
    pairs_path = os.path.join(current_dir, "evaluation_pairs", "evaluation_pairs_all_bounding_box_test.txt")

    image_pairs = load_image_pairs(pairs_path)
    if not image_pairs:
        print("❌ No image pairs loaded. Please check the file path and contents.")
        exit()

    # Now call the evaluate_model function after the check
    fpr, tpr, auc_score, avg_nn_score = evaluate_model(image_pairs, dataset_path)  # This will now be executed
    print(f"AUC Score: {auc_score}")
