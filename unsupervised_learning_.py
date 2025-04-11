import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from tqdm import tqdm

# ---------------------- Pose Dataset from Pickle ---------------------- #
class PicklePoseDataset(Dataset):
    def __init__(self, pkl_file_path):
        with open(pkl_file_path, 'rb') as f:
            self.data = pickle.load(f)
        
        self.samples = []
        for entry in self.data:
            landmarks = entry['landmarks']  # shape: [33, 3]
            if landmarks and len(landmarks) == 33:
                self.samples.append(np.array(landmarks, dtype=np.float32).flatten())  # shape: (99,)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx])


# ---------------------- Autoencoder Model ---------------------- #
class PoseAutoencoder(nn.Module):
    def __init__(self, input_size=99, embedding_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_size)
        )

    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed, z


# ---------------------- Training ---------------------- #
def train_autoencoder(pkl_path, model_save_path="pose_autoencoder.pth", epochs=50, batch_size=32):
    dataset = PicklePoseDataset(pkl_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = PoseAutoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), model_save_path)
    print(f"✅ Model saved to {model_save_path}")


# ---------------------- Main ---------------------- #
if __name__ == "__main__":
    train_autoencoder("pose_landmarks_unsupervised_dataset.pkl")  # <- your .pkl file
