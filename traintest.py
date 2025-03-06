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
    
    # Iterate through all the triplets and extract the landmarks
    for triplet in triplets:
        anchor_landmarks, positive_landmarks, negative_landmarks = triplet
        
        # Extract the coordinates for each of the landmarks
        anchor = extract_landmarks(anchor_landmarks)
        positive = extract_landmarks(positive_landmarks)
        negative = extract_landmarks(negative_landmarks)
        
        # Append to the corresponding lists
        anchors.append(anchor)
        positives.append(positive)
        negatives.append(negative)
    
    # Convert lists into numpy arrays or PyTorch tensors
    anchors = np.array(anchors)
    positives = np.array(positives)
    negatives = np.array(negatives)
    
    # If you're using PyTorch, convert to tensors
    anchors_tensor = torch.tensor(anchors, dtype=torch.float32)
    positives_tensor = torch.tensor(positives, dtype=torch.float32)
    negatives_tensor = torch.tensor(negatives, dtype=torch.float32)
    
    return anchors_tensor, positives_tensor, negatives_tensor

# 4. Training Loop
def train_triplet_loss_model(model, anchors_tensor, positives_tensor, negatives_tensor, optimizer, num_epochs=10):
    model.train()  # Set model to training mode
    
    for epoch in range(num_epochs):
        # Flatten the anchors, positives, and negatives (from [N, M, 3] to [N, 99])
        anchors_tensor = anchors_tensor.view(anchors_tensor.size(0), -1)  # Flatten to [N, 99]
        positives_tensor = positives_tensor.view(positives_tensor.size(0), -1)  # Flatten to [N, 99]
        negatives_tensor = negatives_tensor.view(negatives_tensor.size(0), -1)  # Flatten to [N, 99]
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        anchor_embeddings = model(anchors_tensor)
        positive_embeddings = model(positives_tensor)
        negative_embeddings = model(negatives_tensor)
        
        # Compute triplet loss
        loss = triplet_loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
        
        # Backward pass
        loss.backward()
        
        # Update model parameters
        optimizer.step()
        
        # Print the loss for this epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")
        
    print("Training completed successfully!")
# 5. Main Execution: Define and Train the Model
if __name__ == "__main__":
    # Example usage:
    # 1. Load triplet data
    anchors_tensor, positives_tensor, negatives_tensor = load_triplets('triplets.pkl')
    
    # 2. Create model and loss function
    model = TripletLossModel(input_size=99, embedding_size=128)
    triplet_loss_fn = TripletLoss(margin=1.0)
    
    # 3. Define an optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 4. Train the model
    train_triplet_loss_model(model, anchors_tensor, positives_tensor, negatives_tensor, optimizer, num_epochs=10)
