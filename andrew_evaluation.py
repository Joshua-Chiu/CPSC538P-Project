import torch
import ast
import cv2
import mediapipe as mp
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
from train_triplets import PoseEmbeddingNet  # Import your model definition file

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load the pre-trained pose embedding model
pose_embedding_model = PoseEmbeddingNet(input_size=99, embedding_size=128)  # Adjust according to your model
pose_embedding_model.load_state_dict(torch.load('pose_embedding_model_fine_tuned.pth', weights_only=True))
pose_embedding_model.eval()  # Set the model to evaluation mode

# Function to extract pose landmarks from an image
def extract_pose_landmarks(image_path):
    image = cv2.imread(image_path)
    
    # Check if the image is loaded properly
    if image is None:
        print(f"Error loading image: {image_path}")
        return None  # Skip processing if the image couldn't be loaded

    # Convert the image to RGB before passing it to MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and extract pose landmarks
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z])
        return torch.tensor(landmarks).flatten()  # Flatten the landmarks to 1D tensor
    else:
        print(f"No pose landmarks detected in {image_path}")
        return None

# Function to evaluate the model on a list of image pairs
def evaluate_model(image_pairs):
    true_labels = []
    predicted_scores = []
    skipped_files = 0  # Counter for skipped files
    total_files = len(image_pairs)
    correct_positive = 0
    correct_negative = 0

    all_embeddings = []  # To store embeddings for t-SNE and nearest neighbor accuracy
    all_labels = []  # To store true labels for t-SNE visualization

    for img1_path, img2_path, label in image_pairs:
        print(f"Processing: {img1_path} and {img2_path}")
        # Extract pose landmarks from both images
        landmarks1 = extract_pose_landmarks(img1_path)
        landmarks2 = extract_pose_landmarks(img2_path)

        # Skip pairs where landmarks could not be extracted
        if landmarks1 is None or landmarks2 is None:
            skipped_files += 1
            continue

        # Get embeddings for both images using the pose embedding model
        embedding1 = pose_embedding_model(landmarks1.unsqueeze(0))  # Add batch dimension
        embedding2 = pose_embedding_model(landmarks2.unsqueeze(0))  # Add batch dimension

        # Store the embeddings and labels for t-SNE and nearest neighbor accuracy
        all_embeddings.append(embedding1.detach().numpy())  # Convert to numpy for t-SNE
        all_embeddings.append(embedding2.detach().numpy())
        all_labels.append(label)
        all_labels.append(label)

        # Compute the similarity between embeddings (e.g., using cosine similarity)
        similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2)

        # Append true label and predicted score for ROC curve
        true_labels.append(label)
        predicted_scores.append(similarity.item())

        # Check if the prediction matches the true label
        if (similarity.item() > 0.5 and label == 1) or (similarity.item() <= 0.5 and label == 0):
            if label == 1:
                correct_positive += 1
            else:
                correct_negative += 1

    # Print number of skipped files
    print(f"Skipped {skipped_files} files out of {total_files} due to no landmarks detected.")

    # Calculate ROC curve and AUC score
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_scores)
    auc_score = auc(fpr, tpr)

    # Print the breakdown of correct positive/negative predictions
    print(f"Correctly identified {correct_positive} positive pairs.")
    print(f"Correctly identified {correct_negative} negative pairs.")
    print(f"Total true positive pairs: {true_labels.count(1)}")
    print(f"Total true negative pairs: {true_labels.count(0)}")

    # Apply t-SNE to reduce the dimensionality of the embeddings
    all_embeddings = np.vstack(all_embeddings)  # Stack embeddings into a single array
    tsne = TSNE(n_components=2, random_state=42)  # Reduce to 2D
    tsne_results = tsne.fit_transform(all_embeddings)

    # Plot t-SNE results
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=all_labels, cmap='coolwarm', alpha=0.7)
    plt.colorbar(scatter, label='True Label')
    plt.title('t-SNE Visualization of Pose Embeddings')
    plt.xlabel('t-SNE component 1')
    plt.ylabel('t-SNE component 2')
    plt.show()

    # Nearest Neighbor Accuracy
    embeddings_array = np.vstack(all_embeddings)  # Convert embeddings to numpy array
    dist_matrix = cdist(embeddings_array, embeddings_array, metric='cosine')  # Cosine distance matrix
    np.fill_diagonal(dist_matrix, np.inf)  # Remove self-comparison by setting diagonal to infinity

    # Find nearest neighbor for each sample
    nearest_neighbors = np.argmin(dist_matrix, axis=1)

    # Compute nearest neighbor accuracy
    correct_nn = 0
    total_nn = 0
    for i in range(len(all_labels)):
        nearest_neighbor_label = all_labels[nearest_neighbors[i]]
        if nearest_neighbor_label == all_labels[i]:
            correct_nn += 1
        total_nn += 1

    nn_accuracy = correct_nn / total_nn
    print(f"Nearest Neighbor Accuracy: {nn_accuracy * 100:.2f}%")

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.show()

    return fpr, tpr, auc_score

def load_image_pairs(filepath):
    pairs = []
    with open(filepath, 'r') as f:
        for line in f:
            try:
                # Safely evaluate the tuple string to a Python tuple
                img1, img2, label = ast.literal_eval(line.strip())
                pairs.append((img1.strip(), img2.strip(), int(label)))
            except (ValueError, SyntaxError):
                print(f"⚠️ Skipping malformed line: {line.strip()}")
    return pairs


image_pairs = load_image_pairs("evaluation_pairs.txt")

if not image_pairs:
    print("❌ No image pairs loaded. Please check the file path and contents.")
    exit()

fpr, tpr, auc_score = evaluate_model(image_pairs)

# Print the AUC score
print(f"AUC Score: {auc_score}")
