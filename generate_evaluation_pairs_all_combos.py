import os
import random
from itertools import combinations
from collections import defaultdict
import cv2

def create_image_pairs(dataset_path):
    person_to_images = defaultdict(list)

    # Group images by person ID (prefix before the first underscore)
    for filename in os.listdir(dataset_path):
        if filename.endswith(".png"):  # Adjusted for PNG files as per your dataset
            person_id = filename.split("_")[0]
            image_path = os.path.join(dataset_path, filename)
            person_to_images[person_id].append(image_path)

    image_pairs = []
    positive_pairs_count = defaultdict(int)
    negative_pairs_count = defaultdict(int)
    total_positive = 0
    total_negative = 0

    # Generate all positive pairs
    for person_id, images in person_to_images.items():
        if len(images) < 2:
            continue  # Can't form pairs with fewer than 2 images

        all_pos_combinations = list(combinations(images, 2))
        total_positive += len(all_pos_combinations)

        for img1, img2 in all_pos_combinations:
            image_pairs.append((img1, img2, 1))
            positive_pairs_count[person_id] += 1  # Count positive pairs for the person

    # Generate negative pairs to match the total number of positive pairs
    person_ids = list(person_to_images.keys())
    while total_negative < total_positive:
        person_id = random.choice(person_ids)
        if len(person_to_images[person_id]) < 1:
            continue

        # Pick a random other person who has at least one image
        other_id = random.choice([pid for pid in person_ids if pid != person_id and len(person_to_images[pid]) > 0])
        img1 = random.choice(person_to_images[person_id])
        img2 = random.choice(person_to_images[other_id])

        image_pairs.append((img1, img2, 0))
        negative_pairs_count[person_id] += 1  # Count negative pairs for the person
        total_negative += 1

    random.shuffle(image_pairs)

    # Debug: Print counts of positive and negative pairs for each individual
    print("Positive and Negative Pair Counts per Individual:")
    for person_id in person_to_images.keys():
        print(f"Person {person_id} - Positive pairs: {positive_pairs_count[person_id]}, Negative pairs: {negative_pairs_count[person_id]}")

    # Summary
    print("\nSummary:")
    print(f"Total positive pairs generated: {total_positive}")
    print(f"Total negative pairs generated: {total_negative}")
    print(f"Total image pairs: {len(image_pairs)}\n")

    return image_pairs

# Example usage
if __name__ == "__main__":
    # Path to the directory where the script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Build the dataset path relative to this script
    dataset_path = os.path.join(current_dir, "dataset_ETHZ", "seq1")

    # Generate the pairs
    pairs = create_image_pairs(
        dataset_path
    )

    # Save output file in the same directory
    output_file = os.path.join(current_dir, "evaluation_pairs", "evaluation_pairs_all_combos_seq1.txt")
    with open(output_file, "w") as f:
        for pair in pairs:
            f.write(f"{pair}\n")
    print(f"✅ Evaluation pairs saved to {output_file}")

    # Print a few debug details
    labels = [label for _, _, label in pairs]
    print(f"\nUnique labels in generated pairs: {set(labels)}")
    for i in range(min(10, len(pairs))):
        print(pairs[i]) 
