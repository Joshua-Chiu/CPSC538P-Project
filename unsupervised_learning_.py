import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import mediapipe as mp
import cv2
import numpy as np
from tqdm import tqdm

# ---------------------- Pose Dataset ---------------------- #
class PoseDataset(Dataset):
    def __init__(self, folder_path):
        self.image_paths = [os.path.join(folder_path, f)
                            for f in os.listdir(folder_path)
                            if f.endswith((".png", ".jpg"))]
        self.pose = mp.solutions.pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9)

    def __len__(self):
        return len(self.image_paths)

    def extract_pose(self, image_path):
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
            return coords.flatten()  # Flatten to (33*3,)
        else:
            return None

    def __getitem__(self, idx):
        while True:
            coords = self.extract_pose(self.image_paths[idx])
            if coords is not None:
                return torch.tensor(coords)
            idx = (idx + 1) % len(self)

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
def train_autoencoder(dataset_path, model_save_path="pose_autoencoder.pth", epochs=50, batch_size=32):
    dataset = PoseDataset(dataset_path)
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
    print("Model saved.")

# ---------------------- Main ---------------------- #
if __name__ == "__main__":
    train_autoencoder("entireid/bounding_box_test")  # <- change to your dataset path
