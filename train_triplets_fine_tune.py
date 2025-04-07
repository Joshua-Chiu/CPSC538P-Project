import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np

# PoseEmbeddingNet model (with dropout added)
class PoseEmbeddingNet(nn.Module):
    def __init__(self, input_size=99, embedding_size=128, dropout_rate=0.5):
        super(PoseEmbeddingNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, embedding_size)
        self.dropout = nn.Dropout(p=dropout_rate)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = F.relu(self.fc2(x))
        x = self.dropout(x)  # Apply dropout
        x = self.fc3(x)
        return x

# Triplet loss function
def triplet_loss(anchor, positive, negative, margin=1.0):
    pos_dist = torch.norm(anchor - positive, p=2, dim=1)
    neg_dist = torch.norm(anchor - negative, p=2, dim=1)
    loss = torch.clamp(pos_dist - neg_dist + margin, min=0.0)
    return loss.mean()

# 🛑 Prevent training logic from running on import
if __name__ == "__main__":
    # Load the triplet tensors
    triplet_tensors = torch.load("triplet_tensors.pt")
    print(f"Loaded data type: {type(triplet_tensors)}")
    print(f"Sample of loaded data (first element): {triplet_tensors[0] if isinstance(triplet_tensors, list) else triplet_tensors}")

    if isinstance(triplet_tensors, dict):
        anchors_tensor = triplet_tensors['anchors']
        positives_tensor = triplet_tensors['positives']
        negatives_tensor = triplet_tensors['negatives']
    elif isinstance(triplet_tensors, list) and len(triplet_tensors) == 3:
        anchors_tensor, positives_tensor, negatives_tensor = triplet_tensors
    else:
        raise ValueError("The loaded tensor data format is incorrect!")

    if not isinstance(anchors_tensor, torch.Tensor):
        anchors_tensor = torch.tensor(anchors_tensor, dtype=torch.float32)
    if not isinstance(positives_tensor, torch.Tensor):
        positives_tensor = torch.tensor(positives_tensor, dtype=torch.float32)
    if not isinstance(negatives_tensor, torch.Tensor):
        negatives_tensor = torch.tensor(negatives_tensor, dtype=torch.float32)

    dataset = TensorDataset(anchors_tensor, positives_tensor, negatives_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Model, optimizer, and learning rate scheduler
    model = PoseEmbeddingNet(input_size=99, embedding_size=128, dropout_rate=0.5)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    num_epochs = 10
    best_val_loss = np.inf  # Store the best validation loss for early stopping
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (anchor, positive, negative) in enumerate(dataloader):
            optimizer.zero_grad()
            anchor_embed = model(anchor)
            positive_embed = model(positive)
            negative_embed = model(negative)
            loss = triplet_loss(anchor_embed, positive_embed, negative_embed)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        scheduler.step()  # Update the learning rate

        # Print training loss
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader)}")

    # Save model after training
    torch.save(model.state_dict(), "pose_embedding_model_fine_tuned.pth")
    print("✅ Model training completed and saved as 'pose_embedding_model_fine_tuned.pth'")
