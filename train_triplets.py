import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define the Pose Embedding Network
class PoseEmbeddingNet(nn.Module):
    def __init__(self, input_size=99, embedding_size=128):
        super().__init__()
        self.input_size = input_size  # Use the input size to define the first layer
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 256),  # Use input_size as the input dimension
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_size)
        )

    def forward(self, x):
        return self.network(x)

# Define Triplet Loss
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        self.loss_fn = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, positive, negative):
        return self.loss_fn(anchor, positive, negative)

# Function to train the model
def train_model():
    # Load triplet tensors
    data = torch.load("triplet_tensors.pt")
    anchors = data['anchors']
    positives = data['positives']
    negatives = data['negatives']

    # Prepare Dataloader
    dataset = TensorDataset(anchors, positives, negatives)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Setup training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PoseEmbeddingNet().to(device)
    loss_fn = TripletLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    epochs = 20
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for anchor, positive, negative in loader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            optimizer.zero_grad()
            emb_anchor = model(anchor)
            emb_positive = model(positive)
            emb_negative = model(negative)

            loss = loss_fn(emb_anchor, emb_positive, emb_negative)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(loader):.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "pose_embedding_model.pth")
    print("✅ Pose embedding model saved to pose_embedding_model.pth")

# Only run the training when the script is executed directly
if __name__ == "__main__":
    train_model()
