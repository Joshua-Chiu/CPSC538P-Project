import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

# Define the Pose Embedding Network (same as before)
class PoseEmbeddingNet(nn.Module):
    def __init__(self, input_size=99, embedding_size=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),  
            nn.Linear(input_size, 512),  
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),  
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, embedding_size)  
        )

    def forward(self, x):
        return self.network(x)


class TripletLoss(nn.Module):
    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin
        self.loss_fn = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, positive, negative):
        return self.loss_fn(anchor, positive, negative)

def train_model(margin=2.0, epochs=50):
    data = torch.load("triplet_tensors.pt")
    anchors = data['anchors']
    positives = data['positives']
    negatives = data['negatives']
    
    # Flatten the tensors to make them 2D (batch_size, num_features)
    anchors_flattened = anchors.view(anchors.size(0), -1)
    positives_flattened = positives.view(positives.size(0), -1)
    negatives_flattened = negatives.view(negatives.size(0), -1)

    # Optionally, scale data (normalize embeddings before training)
    scaler = StandardScaler()
    anchors_normalized = torch.tensor(scaler.fit_transform(anchors_flattened.numpy()))
    positives_normalized = torch.tensor(scaler.transform(positives_flattened.numpy()))
    negatives_normalized = torch.tensor(scaler.transform(negatives_flattened.numpy()))

    # Reshape back to original dimensions if necessary
    anchors = anchors_normalized.view(anchors.size(0), anchors.size(1), -1)
    positives = positives_normalized.view(positives.size(0), positives.size(1), -1)
    negatives = negatives_normalized.view(negatives.size(0), negatives.size(1), -1)
    
    # Prepare the DataLoader
    dataset = TensorDataset(anchors, positives, negatives)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PoseEmbeddingNet().to(device)
    loss_fn = TripletLoss(margin=margin)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    patience = 5
    best_loss = float('inf')
    epochs_without_improvement = 0
    
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

        # Update scheduler
        scheduler.step(total_loss)

        if total_loss < best_loss:
            best_loss = total_loss
            epochs_without_improvement = 0
            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(loader):.4f}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping after {epoch+1} epochs")
            break
    
    # Save the trained model
    torch.save(model.state_dict(), "pose_embedding_model.pth")
    print("✅ Pose embedding model saved")

if __name__ == "__main__":
    train_model(margin=2.0, epochs=50)
