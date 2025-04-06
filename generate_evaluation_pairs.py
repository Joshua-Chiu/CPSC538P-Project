import os
import random
from itertools import combinations
from collections import defaultdict

def create_image_pairs(dataset_path, max_positive_pairs_per_id=5, num_negative_pairs_per_id=5):
    person_to_images = defaultdict(list)

    # Group images by person ID (prefix before the first underscore)
    for filename in os.listdir(dataset_path):
        if filename.endswith(".jpg"):  # Adjusted for JPG files as per your dataset
            person_id = filename.split("_")[0]
            person_to_images[person_id].append(os.path.join(dataset_path, filename))

    image_pairs = []
    positive_pairs_count = defaultdict(int)
    negative_pairs_count = defaultdict(int)
    total_positive = 0
    total_negative = 0

    # Generate positive pairs
    for person_id, images in person_to_images.items():
        if len(images) < 2:
            continue  # Can't form pairs with fewer than 2 images

        all_pos_combinations = list(combinations(images, 2))
        random.shuffle(all_pos_combinations)

        num_to_sample = min(max_positive_pairs_per_id, len(all_pos_combinations))
        selected_pairs = all_pos_combinations[:num_to_sample]

        for img1, img2 in selected_pairs:
            image_pairs.append((img1, img2, 1))
            positive_pairs_count[person_id] += 1  # Count positive pairs for the person
            total_positive += 1

    # Generate negative pairs
    person_ids = list(person_to_images.keys())
    for person_id in person_to_images:
        for _ in range(num_negative_pairs_per_id):
            if len(person_to_images[person_id]) == 0:
                continue

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
    # Get the path to the current file
    current_dir = os.path.dirname(__file__)
    dataset_path = os.path.join(current_dir, "entireid", "bounding_box_test")

    pairs = create_image_pairs(
        dataset_path,
        max_positive_pairs_per_id=5,  # Adjustable
        num_negative_pairs_per_id=5   # Adjustable
    )

    # Print sample pairs
    for i in range(min(10, len(pairs))):
        print(pairs[i])
