

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import librosa
from sklearn.model_selection import train_test_split
DATA_DIR = "data"
DATASET_SLUG = "chrisfilo/urbansound8k"
DATA_ROOT = os.path.join(DATA_DIR, "UrbanSound8K")
CSV_PATH = os.path.join(DATA_ROOT, "metadata", "UrbanSound8K.csv")
AUDIO_DIR = os.path.join(DATA_ROOT, "audio")
SAMPLE_RATE = 22050
DURATION = 4
N_MELS = 64
NUM_CLASSES = 10
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3


def download_dataset():
    if os.path.exists(CSV_PATH):
        print(f"Dataset already present at {DATA_ROOT}. Skipping download.")
        return
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        print("Downloading UrbanSound8K from Kaggle...")
        os.makedirs(DATA_DIR, exist_ok=True)
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(DATASET_SLUG, path=DATA_DIR, unzip=True)
        print("Download complete.")
    except Exception as e:
        print("Download failed. Ensure ~/.kaggle/kaggle.json exists and you have accepted the dataset rules on Kaggle.")
        raise e


class UrbanSoundDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(AUDIO_DIR, f"fold{row['fold']}", row["slice_file_name"])
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        target_len = SAMPLE_RATE * DURATION
        if len(y) < target_len:
            y = librosa.util.fix_length(y, target_len)
        else:
            y = y[:target_len]
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        x = torch.tensor(mel_db, dtype=torch.float32)
        x = x.unsqueeze(0)
        label = int(row["classID"])
        return x, label


def _conv_out_size(n_mels, time_frames, pool_steps=2):
    for _ in range(pool_steps):
        n_mels = (n_mels + 1) // 2
        time_frames = (time_frames + 1) // 2
    return 32 * n_mels * time_frames


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, n_mels=N_MELS, sr=SAMPLE_RATE, duration=DURATION):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        time_frames = (sr * duration) // 512 + 1
        fc_in = _conv_out_size(n_mels, time_frames)
        self.fc1 = nn.Linear(fc_in, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def train():
    download_dataset()
    if not os.path.exists(CSV_PATH):
        print(f"Expected CSV at {CSV_PATH}. Aborting.")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["classID"], random_state=42
    )

    train_ds = UrbanSoundDataset(train_df)
    val_ds = UrbanSoundDataset(val_df)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        avg_loss = total_loss / len(train_loader.dataset)

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = logits.argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        acc = correct / total
        print(f"Epoch {epoch + 1}/{EPOCHS}: loss={avg_loss:.4f}, val_acc={acc:.3f}")

    print("Done.")


if __name__ == "__main__":
    train()
