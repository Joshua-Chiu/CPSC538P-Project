# Final Project Repository for CPSC 538P

## Team Members
- Rowel Sabahat
- Andrew Feng
- Josh Chiu

---

## Project Structure

The folders `entireid` and `dataset_ETHZ` contain the training and evaluation datasets for our project.

---

## Model Training

Our journey begins with the following steps:

1. **`create_dataset_entireid.py`**: 
   - Cleans the original dataset and extracts pose data from the `entireid` dataset using **Google MediaPipe**.

2. **`create_triplets.py`**:
   - Converts the extracted poses into triplets.

3. **`pkl_landmark_to_tensors.py`**:
   - Converts the landmark pickle files into tensors.

4. **`train_triplets.py`**:
   - The tensors are passed through our training model, which uses **Triplet Loss** to output our machine learning model.

---

## Evaluation

Our evaluation process includes the following steps:

1. **`generate_evaluation_pairs.py`**: 
   - Generates positive and negative image pairs with a ground truth label from all three sequences of the `dataset_ETHZ`.

2. **`main_evaluation.py`**:
   - The image pairs and machine learning model are passed into this file to generate the metrics and graphs necessary for evaluation.

---

Feel free to check the individual files for more details about the implementation and experiment outcomes.
