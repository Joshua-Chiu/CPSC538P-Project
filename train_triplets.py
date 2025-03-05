import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# ==============================
# 1️⃣ Load and Preprocess Triplet Data
# ==============================

# Load the pose data from the pickle file
def load_pose(file_path):
    with open(file_path, 'rb') as f:
        pose = pickle.load(f)

    # Assuming pose is a list of Landmark objects and each Landmark has x, y, z attributes
    keypoints = np.array([[kp.x, kp.y, kp.z] for kp in pose], dtype=np.float32)  # Extract x, y, z coordinates
    
    return keypoints

# Load triplets from the pickle file
with open("triplets.pkl", "rb") as f:
    triplets = pickle.load(f)  # triplets should be a list of (anchor, positive, negative)

# Validate triplet data
if not triplets or not isinstance(triplets, list):
    raise ValueError("Error: 'triplets.pkl' does not contain valid triplet data.")

# Convert triplets into a NumPy array for easier manipulation
triplets = np.array(triplets, dtype=object)  # dtype=object to avoid shape issues

# Detect available device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom Dataset Class for PyTorch
class TripletDataset(Dataset):
    """
    Custom Dataset class for loading triplet data in PyTorch DataLoader.
    Each item consists of (anchor, positive, negative) pose embeddings.
    """
    def __init__(self, triplets):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, index):
        anchor, positive, negative = self.triplets[index]

        # Load and process the poses
        anchor = load_pose(anchor)  # Load pose for anchor
        positive = load_pose(positive)  # Load pose for positive
        negative = load_pose(negative)  # Load pose for negative

        # Ensure NumPy array conversion with proper dtype
        anchor = torch.tensor(anchor, dtype=torch.float32, device=device)
        positive = torch.tensor(positive, dtype=torch.float32, device=device)
        negative = torch.tensor(negative, dtype=torch.float32, device=device)

        return anchor, positive, negative

# Define DataLoader to handle batch processing
train_loader = DataLoader(TripletDataset(triplets), batch_size=32, shuffle=True)

# ==============================
# 2️⃣ Define the Neural Network
# ==============================

class PoseEmbeddingNet(nn.Module):
    """
    Neural network that converts input pose keypoints into an embedding space.
    This is a simple feedforward network (MLP).
    """
    def __init__(self, input_dim=34, embedding_dim=128):  # Adjust input_dim based on pose data size
        super(PoseEmbeddingNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),  # First hidden layer (256 neurons)
            nn.ReLU(),
            nn.Linear(256, 128),  # Second hidden layer (128 neurons)
            nn.ReLU(),
            nn.Linear(128, embedding_dim)  # Output embedding (128-dimensional)
        )

    def forward(self, x):
        return self.fc(x)

# Instantiate the model and move it to the detected device
model = PoseEmbeddingNet().to(device)

# ==============================
# 3️⃣ Define Loss Function & Optimizer
# ==============================

# Use PyTorch's built-in Triplet Margin Loss function
triplet_loss = nn.TripletMarginLoss(margin=1.0)  # Margin controls how far apart negatives should be

# Define an optimizer (Adam is commonly used for deep learning tasks)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==============================
# 4️⃣ Train the Model
# ==============================

# Number of training epochs (adjust as needed)
num_epochs = 20  

for epoch in range(num_epochs):
    total_loss = 0  # Track total loss per epoch
    
    for anchor, positive, negative in train_loader:
        optimizer.zero_grad()  # Reset gradients to avoid accumulation

        # Forward pass: Generate embeddings for anchor, positive, and negative samples
        anchor_emb = model(anchor)
        positive_emb = model(positive)
        negative_emb = model(negative)

        # Compute triplet loss
        loss = triplet_loss(anchor_emb, positive_emb, negative_emb)
        loss.backward()  # Backpropagation to compute gradients
        optimizer.step()  # Update model parameters

        total_loss += loss.detach().cpu().item()  # Accumulate loss

    # Print loss for the current epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")

# ==============================
# 5️⃣ Save and Evaluate the Model
# ==============================

# Save the trained model weights to a file
torch.save(model.state_dict(), "pose_triplet_model.pth")
print("✅ Model saved successfully as 'pose_triplet_model.pth'")

# Load model for evaluation or inference
model.load_state_dict(torch.load("pose_triplet_model.pth", map_location=device))
model.eval()  # Set model to evaluation mode (disables dropout, batchnorm updates)

print("✅ Model loaded and ready for evaluation")
