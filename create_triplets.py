import os
import random
import pickle
import numpy as np
from collections import defaultdict

# Load the pose data from the pickle file
def load_pose(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# Generate triplets: anchor, positive, negative
def generate_triplets(dataset_path, output_file="triplets_test.pkl"):
    triplets = []
    person_images = defaultdict(list)

    # Step 1: Group images by individual based on prefix (before first underscore)
    print(f"Scanning files in {dataset_path}...")  # Debug
    for file_name in sorted(os.listdir(dataset_path)):
        if file_name.endswith(".pkl"):  # Only consider .pkl files
            print(f"Found file: {file_name}")  # Debug
            person_id = file_name.split("_")[0]  # The prefix is the individual ID
            person_images[person_id].append(os.path.join(dataset_path, file_name))

    # Debug: Print how many images each individual has
    print("Images per individual:")
    for person_id, images in person_images.items():
        print(f"{person_id}: {len(images)} images")

    # Step 2: Create triplets
    person_ids = list(person_images.keys())
    print(f"Found {len(person_ids)} individuals.")  # Debug

    for person_id in person_ids:
        images = person_images[person_id]
        
        # Skip if this individual has fewer than 2 images
        if len(images) < 2:
            print(f"Skipping individual {person_id} due to insufficient images.")
            continue
        
        # The first image is the anchor, the second is the positive
        anchor_path = images[0]  # First image as the anchor
        positive_path = images[1]  # Second image as the positive
        
        # Select a negative from a different individual
        negative_person_id = random.choice([pid for pid in person_ids if pid != person_id])
        negative_path = random.choice(person_images[negative_person_id])  # Random negative image
        
        # Load the poses from the pickle files
        anchor = load_pose(anchor_path)
        positive = load_pose(positive_path)
        negative = load_pose(negative_path)
        
        # Append the triplet to the list
        triplets.append((anchor, positive, negative))
    
    # Step 3: Save the triplets to a .pkl file
    with open(output_file, "wb") as f:
        pickle.dump(triplets, f)
    
    print(f"✅ Saved {len(triplets)} triplets to {output_file}")
    return np.array(triplets)

# Define dataset path (adjust this based on your file structure)
dataset_path = "entireid/bounding_box_test_pose"  # Adjust this path if necessary_test/test

# Generate triplets and save to file
triplets = generate_triplets(dataset_path, "triplets_test.pkl")
