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
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from train_triplets import PoseEmbeddingNet
import time

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
    with ProcessPoolExecutor() as executor:
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

def evaluate_model(image_pairs, dataset_path, batch_size=5000, is_training=True):
    start_time = time.time()
    dataset_name = os.path.basename(dataset_path.rstrip("/\\"))
    true_labels = []
    predicted_scores = []
    skipped_files = 0
    total_files = len(image_pairs)
    correct_positive = 0
    correct_negative = 0
    total_true_positives = 0  # For counting ground truth positives
    total_true_negatives = 0  # For counting ground truth negatives
    all_embeddings = []
    all_labels = []

    # Extract landmarks for all images in pairs
    all_imgpaths = list(set([p for pair in image_pairs for p in pair[:2]]))
    pose_dict = dict(zip(all_imgpaths, batch_extract_landmarks(all_imgpaths)))

    for idx, (img1path, img2path, label) in enumerate(tqdm(image_pairs, desc="Evaluating Pairs", unit="pair")):
        landmarks1 = pose_dict.get(img1path)
        landmarks2 = pose_dict.get(img2path)

        if landmarks1 is None or landmarks2 is None:
            # print(f"⚠️ Skipped pair due to missing landmarks: {img1path}, {img2path}")
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

        # Calculate the total number of true positive pairs (ground truth label == 1)
        if label == 1:  # If the ground truth label is 1 (same person)
            total_true_positives += 1
        
        # Calculate the total number of true negative pairs (ground truth label == 0)
        if label == 0:  # If the ground truth label is 0 (different person)
            total_true_negatives += 1

        # Model Prediction for True Positive (TP) and True Negative (TN) Calculation
        if similarity.item() > 0.5:  # Predicted as same person
            if label == 1:  # Correctly predicted same person
                correct_positive += 1
        else:  # Predicted as different person
            if label == 0:  # Correctly predicted different person
                correct_negative += 1

        if idx % 1000 == 0 and idx > 0:
            elapsed_time = time.time() - start_time
            avg_time_per_pair = elapsed_time / idx
            remaining_pairs = total_files - idx
            estimated_time = avg_time_per_pair * remaining_pairs
            print(f"Processed {idx}/{total_files} pairs. Estimated time remaining: {estimated_time/60:.2f} minutes.")

    total_time = time.time() - start_time
    print(f"✅ Finished processing all image pairs. Total time: {total_time/60:.2f} minutes.")

    if not all_embeddings:
        print("❌ No valid embeddings were generated. Exiting early.")
        return None, None, None

    # t-SNE visualization
    print("🚦 Starting t-SNE visualization...")

    all_embeddings = np.vstack(all_embeddings)

    # Sample a smaller subset for t-SNE
    sample_size = min(10000, len(all_embeddings))
    indices = np.random.choice(len(all_embeddings), size=sample_size, replace=False)
    sampled_embeddings = all_embeddings[indices]
    sampled_labels = [all_labels[i] for i in indices]

    # Estimate ETA for t-SNE
    print("⏳ Estimating t-SNE time with 100 samples...")
    tsne_sample = TSNE(n_components=2, random_state=42, perplexity=30)
    start_tsne_sample = time.time()
    tsne_sample.fit_transform(sampled_embeddings[:100])
    end_tsne_sample = time.time()
    estimated_tsne_time = (end_tsne_sample - start_tsne_sample) * (sample_size /100)
    print(f"🕒 Estimated t-SNE time for {sample_size} embeddings: {estimated_tsne_time:.2f} seconds.")

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_results = tsne.fit_transform(sampled_embeddings)

    print("🌀 t-SNE computation finished. Plotting now...")

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=sampled_labels, cmap='coolwarm', alpha=0.7)
    plt.colorbar(scatter, label='True Label')
    plt.title('t-SNE Visualization of Pose Embeddings')
    plt.xlabel('t-SNE component 1')
    plt.ylabel('t-SNE component 2')
    tsne_filename = f"{dataset_name}_tsne.png"
    plt.savefig(tsne_filename)
    print(f"📸 t-SNE plot saved as {tsne_filename}")
    plt.close()

    # ROC Curve
    true_labels = np.array(true_labels)
    predicted_scores = np.array(predicted_scores)

    # Binning the predicted scores for faster ROC computation
    bins = np.linspace(0, 1, num=200)  # You can try 100 for even faster runs
    digitized_scores = np.digitize(predicted_scores, bins) / len(bins)

    # ROC curve computation
    fpr, tpr, thresholds = roc_curve(true_labels, digitized_scores)
    auc_score = auc(fpr, tpr)

    print(f"Correctly identified {correct_positive} positive pairs.")
    print(f"Correctly identified {correct_negative} negative pairs.")
    print(f"Total true positives (ground truth positives): {total_true_positives}")
    print(f"Total true negatives (ground truth negatives): {total_true_negatives}")

    return fpr, tpr, auc_score

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path_train = os.path.join(current_dir, "entireid", "bounding_box_test")
    dataset_path_test = os.path.join(current_dir, "dataset_ETHZ", "seq2")
    pairs_path_train = os.path.join(current_dir, "evaluation_pairs", "evaluation_pairs_all_combos_bounding_box_test.txt")
    pairs_path_test = os.path.join(current_dir, "evaluation_pairs", "evaluation_pairs_all_combos_seq2.txt")

    image_pairs_train = load_image_pairs(pairs_path_train)
    image_pairs_test = load_image_pairs(pairs_path_test)

    if not image_pairs_train or not image_pairs_test:
        print("❌ No image pairs loaded. Please check the file path and contents.")
        exit()

    fpr_train, tpr_train, auc_score_train = evaluate_model(image_pairs_train, dataset_path_train, is_training=True)
    fpr_test, tpr_test, auc_score_test = evaluate_model(image_pairs_test, dataset_path_test, is_training=False)

    # Check for None values before plotting
    if fpr_train is None or tpr_train is None or auc_score_train is None:
        print("❌ Training data evaluation failed. Skipping training ROC curve.")
        fpr_train, tpr_train, auc_score_train = [], [], 0.0

    if fpr_test is None or tpr_test is None or auc_score_test is None:
        print("❌ Testing data evaluation failed. Skipping testing ROC curve.")
        fpr_test, tpr_test, auc_score_test = [], [], 0.0

    # Plot both ROC curves on the same plot
    plt.figure(figsize=(8, 6))
    if fpr_train and tpr_train:
        plt.plot(fpr_train, tpr_train, color='blue', lw=2, label=f'Training ROC curve (AUC = {auc_score_train:.2f})')
    if fpr_test and tpr_test:
        plt.plot(fpr_test, tpr_test, color='red', lw=2, label=f'Testing ROC curve (AUC = {auc_score_test:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.savefig('roc_curves.png')
    print(f"📈 ROC curves saved as roc_curves.png")
    plt.close()

    print(f"AUC Score (Training): {auc_score_train:.4f}")
    print(f"AUC Score (Testing): {auc_score_test:.4f}")