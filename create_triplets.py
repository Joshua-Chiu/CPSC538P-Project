import os
import random
import pickle
import itertools
from collections import defaultdict
import numpy as np

# Load the pose data from the pickle file
def load_pose(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# Generate triplets: anchor, positive, negative
def generate_triplets(dataset_path, output_file="triplets.pkl"):
    triplets = []
    person_images = defaultdict(list)

    # Step 1: Group images by individual based on prefix (before first underscore)
    print(f"Scanning files in {dataset_path}...")
    for file_name in sorted(os.listdir(dataset_path)):
        if file_name.endswith(".pkl"):
            person_id = file_name.split("_")[0]
            person_images[person_id].append(os.path.join(dataset_path, file_name))

    print("Images per individual:")
    for person_id, images in person_images.items():
        print(f"{person_id}: {len(images)} images")

    person_ids = list(person_images.keys())
    print(f"Found {len(person_ids)} individuals.")

    # Step 2: Create all unique anchor-positive combinations for each individual
    for person_id in person_ids:
        images = person_images[person_id]

        if len(images) < 2:
            print(f"Skipping {person_id}: not enough images for pairs.")
            continue

        # All unique pairs (i, j) where i ≠ j
        anchor_positive_pairs = list(itertools.combinations(images, 2))

        for anchor_path, positive_path in anchor_positive_pairs:
            # Randomly select a negative individual ≠ current individual
            negative_person_id = random.choice([pid for pid in person_ids if pid != person_id])
            negative_path = random.choice(person_images[negative_person_id])

            # Load the poses
            anchor = load_pose(anchor_path)
            positive = load_pose(positive_path)
            negative = load_pose(negative_path)

            # Store triplet
            triplets.append((anchor, positive, negative))

    # Step 3: Save the triplets to a .pkl file
    with open(output_file, "wb") as f:
        pickle.dump(triplets, f)

    print(f"✅ Saved {len(triplets)} triplets to {output_file}")
    return np.array(triplets)

# Path to your dataset
dataset_path = "entireid/bounding_box_test_pose"

# Run the triplet generation
triplets = generate_triplets(dataset_path, "triplets.pkl")
