import os
import random
import pickle
import numpy as np

# Load a .pkl file
def load_pose(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# Generate triplets from dataset directory
def generate_triplets(dataset_path, num_triplets=10000):
    triplets = []
    person_folders = [os.path.join(dataset_path, p) for p in os.listdir(dataset_path)]

    for _ in range(num_triplets):
        # Select an individual (Anchor)
        anchor_folder = random.choice(person_folders)
        anchor_files = os.listdir(anchor_folder)

        if len(anchor_files) < 2:
            continue  # Skip if not enough images

        # Pick two different images for Anchor & Positive
        anchor_file, positive_file = random.sample(anchor_files, 2)
        anchor_path = os.path.join(anchor_folder, anchor_file)
        positive_path = os.path.join(anchor_folder, positive_file)

        # Select a different individual (Negative)
        negative_folder = random.choice([f for f in person_folders if f != anchor_folder])
        negative_file = random.choice(os.listdir(negative_folder))
        negative_path = os.path.join(negative_folder, negative_file)

        # Load pose keypoints
        anchor = load_pose(anchor_path)
        positive = load_pose(positive_path)
        negative = load_pose(negative_path)

        triplets.append((anchor, positive, negative))

    return np.array(triplets)

# Example Usage
triplets = generate_triplets("dataset")
print(f"Generated {len(triplets)} triplets.")
