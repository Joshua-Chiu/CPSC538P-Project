import pickle
import numpy as np
import torch

# Function to extract x, y, z coordinates from a list of Landmark objects
def extract_landmarks(landmarks):
    # Handle nested list issue
    if isinstance(landmarks[0], list):
        print("⚠️ Nested list detected! Unpacking first element...")
        landmarks = landmarks[0]  

    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

# Function to load triplets from pickle and convert them into tensors
def load_triplets(file_path):
    with open(file_path, 'rb') as f:
        triplets = pickle.load(f)  # Load the triplets.pkl file

    # Debugging: Print out the structure of one sample triplet
    print(f"Sample triplet structure: {type(triplets[0])}")  # Should be a tuple
    print(f"Sample anchor structure: {type(triplets[0][0])}")  # Should be a list
    print(f"Sample landmark structure: {type(triplets[0][0][0])}")  # Should be a Landmark object

    anchors, positives, negatives = [], [], []

    for anchor_landmarks, positive_landmarks, negative_landmarks in triplets:
        anchors.append(extract_landmarks(anchor_landmarks))
        positives.append(extract_landmarks(positive_landmarks))
        negatives.append(extract_landmarks(negative_landmarks))

    # Convert to PyTorch tensors (optimized conversion to avoid slow warnings)
    anchors_tensor = torch.tensor(np.array(anchors), dtype=torch.float32)
    positives_tensor = torch.tensor(np.array(positives), dtype=torch.float32)
    negatives_tensor = torch.tensor(np.array(negatives), dtype=torch.float32)

    return anchors_tensor, positives_tensor, negatives_tensor

# Load triplets and convert them to tensors
triplets_file = "triplets.pkl"  # Ensure this file exists in your directory
anchors_tensor, positives_tensor, negatives_tensor = load_triplets(triplets_file)

# Save tensors to a .pt file
torch.save({
    'anchors': anchors_tensor,
    'positives': positives_tensor,
    'negatives': negatives_tensor
}, "triplet_tensors.pt")

# Output confirmation
print("✅ Successfully loaded, converted, and saved triplets to triplet_tensors.pt!")
print(f"Anchor tensor shape: {anchors_tensor.shape}")
print(f"Positive tensor shape: {positives_tensor.shape}")
print(f"Negative tensor shape: {negatives_tensor.shape}")
