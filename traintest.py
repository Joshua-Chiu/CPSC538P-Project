import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 1. Model Architecture: TripletLossModel class
class TripletLossModel(nn.Module):
    def __init__(self, input_size=99, embedding_size=128):
        super(TripletLossModel, self).__init__()
        
        # Define the layers of the neural network
        self.fc1 = nn.Linear(input_size, 256)  # 99 input features (33 landmarks * 3 coordinates)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, embedding_size)  # Output embedding size
        
        # Optional: Use ReLU activations after each hidden layer
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 2. Triplet Loss Function
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Compute pairwise distances between anchor and positive, and anchor and negative
        positive_distance = F.pairwise_distance(anchor, positive, p=2)  # Euclidean distance
        negative_distance = F.pairwise_distance(anchor, negative, p=2)
        
        # Calculate triplet loss with margin
        loss = torch.clamp(positive_distance - negative_distance + self.margin, min=0.0)
        return loss.mean()  # Return the mean loss

# 3. Function to load and process the triplet data
def load_triplets(path):
    # Load the triplets data from the pickle file
    with open(path, 'rb') as f:
        triplets = pickle.load(f)
    
    # Function to extract x, y, z coordinates from Landmark objects
    def extract_landmarks(landmarks):
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    
    # Initialize lists to store anchor, positive, and negative samples
    anchors = []
    positives = []
    negatives = []
    
    # Process the single triplet (since the file contains only one)
    anchor_landmarks, positive_landmarks, negative_landmarks = triplets[0]
    
    # Extract the coordinates for each of the landmarks
    anchor = extract_landmarks(anchor_landmarks)
    positive = extract_landmarks(positive_landmarks)
    negative = extract_landmarks(negative_landmarks)
    
    # A
