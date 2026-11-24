import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

PROCESSED_DATA_PATH = os.path.join("data", "processed", "train_FD001_processed.parquet")
MODEL_DIR = os.path.join("models")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "lstm_rul_model.pt")

SEQUENCE_LENGTH = 30  # 30 çevrimlik kayan pencere (sliding window)
BATCH_SIZE = 64
EPOCHS = 25
LEARNING_RATE = 0.001

FEATURE_COLS = [
    "op_setting_1", "op_setting_2", "sensor_2", "sensor_3", "sensor_4",
    "sensor_7", "sensor_8", "sensor_9", "sensor_11", "sensor_12",
    "sensor_13", "sensor_14", "sensor_15", "sensor_17", "sensor_20", "sensor_21"
]

class CMAPSSSequenceDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

def create_sliding_windows(df, seq_len=30):
    sequences, targets = [], []
    
    for _, group in df.groupby("unit_id"):
        features = group[FEATURE_COLS].values
        rul_values = group["RUL"].values
        
        if len(features) >= seq_len:
            for i in range(len(features) - seq_len + 1):
                sequences.append(features[i : i + seq_len])
                targets.append(rul_values[i + seq_len - 1])
                
    return np.array(sequences), np.array(targets)

class UAVTurbofanLSTM(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=64, num_layers=2, dropout=0.2):
        super(UAVTurbofanLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def train_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Eğitim donanımı: {device}")

    df = pd.read_parquet(PROCESSED_DATA_PATH)
    df["RUL"] = df["RUL"].clip(upper=125)  # NASA C-MAPSS Piecewise Linear RUL standardı
    print(f"İşlenmiş veri yüklendi: {df.shape[0]} kayıt.")

    X, y = create_sliding_windows(df, seq_len=SEQUENCE_LENGTH)
    print(f"Sekans matrisi oluşturuldu: {X.shape} (Örnek, Zaman Adımı, Öznitelik)")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_loader = DataLoader(CMAPSSSequenceDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(CMAPSSSequenceDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    model = UAVTurbofanLSTM(input_dim=len(FEATURE_COLS)).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n--- LSTM Model Eğitimi Başlatılıyor ---")
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for seqs, targets in train_loader:
            seqs, targets = seqs.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(seqs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(targets)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for seqs, targets in val_loader:
                seqs, targets = seqs.to(device), targets.to(device)
                preds = model(seqs)
                loss = criterion(preds, targets)
                val_loss += loss.item() * len(targets)

        val_loss /= len(val_loader.dataset)
        val_rmse = np.sqrt(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] - Train Loss: {train_loss:.2f} | Val Loss: {val_loss:.2f} | Val RMSE: {val_rmse:.2f} çevrim")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f"\nEğitim tamamlandı. En başarılı ağırlıklar kaydedildi -> {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()
