import os
import random
import pickle
from collections import defaultdict

# Define dataset path
dataset_path = "entireid_test/test_pose/"  # Update the dataset path here

# Step 1: Group images by individual based on their filename prefix
def group_images_by_person(dataset_path):
    person_images = defaultdict(list)
    
    # Loop through the dataset directory to group images by prefix
    for file_name in os.listdir(dataset_path):
        if file_name.endswith(".jpg") or file_name.endswith(".png"):  # Assuming images are in .jpg or .png format
            person_id = file_name.split("_")[0]  # Extract the person ID from the filename
            person_images[person_id].append(os.path.join(dataset_path, file_name))
    
    return person_images

# Step 2: Generate triplets
def generate_triplets(dataset_path, output_file="triplets.pkl"):
    person_images = group_images_by_person(dataset_path)
    triplets = []

    for person_id, images in person_images.items():
        if len(images) < 2:  # Skip individuals with fewer than 2 images
            continue

        # Define the anchor (first image) and positive (second image) for this person
        anchor = images[0]
        positive = images[1]
        
        # Select a random negative image from a different individual
        random_person_id = random.choice([pid for pid in person_images if pid != person_id])
        negative = random.choice(person_images[random_person_id])
        
        # Append the triplet (anchor, positive, negative)
        triplets.append((anchor, positive, negative))
    
    # Step 3: Save the triplets to a file
    with open(output_file, "wb") as f:
        pickle.dump(triplets, f)
    
    print(f"✅ Saved {len(triplets)} triplets to {output_file}")
    return triplets

# Step 5: Generate triplets
triplets = generate_triplets(dataset_path, "triplets.pkl")
