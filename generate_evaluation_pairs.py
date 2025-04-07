import os
import random
from itertools import combinations
from collections import defaultdict
import mediapipe as mp
import cv2

# Function to check if landmarks are detected in an image
def has_landmarks(image_path):
    # Load image using OpenCV
    image = cv2.imread(image_path)
    
    # Initialize MediaPipe pose detection
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    # Convert image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform pose detection
    results = pose.process(image_rgb)
    
    # Check if landmarks are detected
    return results.pose_landmarks is not None

def create_image_pairs(dataset_path, max_positive_pairs_per_id=5, num_negative_pairs_per_id=5):
    person_to_images = defaultdict(list)
    images_with_landmarks = 0
    images_without_landmarks = 0

    # Group images by person ID (prefix before the first underscore)
    for filename in os.listdir(dataset_path):
        if filename.endswith(".jpg"):  # Adjusted for JPG files as per your dataset
            person_id = filename.split("_")[0]
            image_path = os.path.join(dataset_path, filename)
            person_to_images[person_id].append(image_path)
            
            # Check if landmarks are detected for the image
            if has_landmarks(image_path):
                images_with_landmarks += 1
            else:
                images_without_landmarks += 1

    image_pairs = []
    positive_pairs_count = defaultdict(int)
    negative_pairs_count = defaultdict(int)
    total_positive = 0
    total_negative = 0

    # Generate positive pairs
    for person_id, images in person_to_images.items():
        if len(images) < 2:
            continue  # Can't form pairs with fewer than 2 images

        # Filter images where both have landmarks detected
        valid_images = [img for img in images if has_landmarks(img)]
        
        if len(valid_images) < 2:
            continue  # Can't form pairs if not enough images with landmarks

        all_pos_combinations = list(combinations(valid_images, 2))
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

            # Ensure both images have landmarks detected
            if has_landmarks(img1) and has_landmarks(img2):
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
    
    # Print out the number of images with and without landmarks in the dataset
    print(f"Number of images with landmarks: {images_with_landmarks}")
    print(f"Number of images without landmarks: {images_without_landmarks}")

    return image_pairs

# Example usage
if __name__ == "__main__":
    # Get the path to the current file
    current_dir = os.path.dirname(__file__)
    dataset_path = os.path.join(current_dir, "entireid", "bounding_box_test")

    # Generate the pairs
    pairs = create_image_pairs(
        dataset_path,
        max_positive_pairs_per_id=5,  # Adjustable
        num_negative_pairs_per_id=5   # Adjustable
    )

    # Save pairs to a txt file
    output_file = "evaluation_pairs.txt"
    with open(output_file, "w") as f:
        for pair in pairs:
            f.write(f"{pair}\n")
    print(f"Evaluation pairs saved to {output_file}")

    # Debug: Print unique labels
    labels = [label for _, _, label in pairs]
    print(f"\nUnique labels in generated pairs: {set(labels)}")

    # Print sample pairs for debugging
    for i in range(min(10, len(pairs))):
        print(pairs[i])
