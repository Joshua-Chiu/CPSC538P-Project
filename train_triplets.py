import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# Define the neural network model for feature extraction
class PoseEmbeddingNet(nn.Module):
    def __init__(self, input_size=99, embedding_size=128):
        super(PoseEmbeddingNet, self).__init__()
        # Define layers for the network
        self.fc1 = nn.Linear(input_size, 512)  # First fully connected layer
        self.fc2 = nn.Linear(512, 256)         # Second fully connected layer
        self.fc3 = nn.Linear(256, embedding_size)  # Output layer (embedding)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))  # ReLU activation after first layer
        x = F.relu(self.fc2(x))  # ReLU activation after second layer
        x = self.fc3(x)          # Output embedding
        return x

# Triplet loss function
def triplet_loss(anchor, positive, negative, margin=1.0):
    # Compute the pairwise distances
    pos_dist = torch.norm(anchor - positive, p=2, dim=1)  # Distance between anchor and positive
    neg_dist = torch.norm(anchor - negative, p=2, dim=1)  # Distance between anchor and negative
    
    # Triplet loss formula: max(0, d(a, p) - d(a, n) + margin)
    loss = torch.clamp(pos_dist - neg_dist + margin, min=0.0)
    return loss.mean()

# Load the triplet tensors from the file
triplet_tensors = torch.load("triplet_tensors.pt")
anchors_tensor, positives_tensor, negatives_tensor = triplet_tensors

# Create a DataLoader for the triplet dataset
dataset = TensorDataset(anchors_tensor, positives_tensor, negatives_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize the model, loss function, and optimizer
model = PoseEmbeddingNet(input_size=99, embedding_size=128)  # 99 because 33 joints * 3 (x, y, z)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    total_loss = 0
    for batch_idx, (anchor, positive, negative) in enumerate(dataloader):
        optimizer.zero_grad()  # Zero the gradients
        
        # Forward pass: get embeddings for anchor, positive, and negative
        anchor_embed = model(anchor)
        positive_embed = model(positive)
        negative_embed = model(negative)
        
        # Calculate triplet loss
        loss = triplet_loss(anchor_embed, positive_embed, negative_embed)
        total_loss += loss.item()  # Accumulate loss
        
        # Backward pass: compute gradients
        loss.backward()
        optimizer.step()  # Update the model parameters
        
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader)}")

# Save the trained model
torch.save(model.state_dict(), "pose_embedding_model.pth")
print("✅ Model training completed and saved as 'pose_embedding_model.pth'")
