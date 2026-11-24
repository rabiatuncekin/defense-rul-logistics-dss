import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

MODEL_PATH = os.path.join("models", "lstm_rul_model.pt")
SCALER_PATH = os.path.join("data", "processed", "scaler.pkl")
RAW_DATA_PATH = os.path.join("data", "raw", "train_FD001.txt")

SEQUENCE_LENGTH = 30
FEATURE_COLS = [
    "op_setting_1", "op_setting_2", "sensor_2", "sensor_3", "sensor_4",
    "sensor_7", "sensor_8", "sensor_9", "sensor_11", "sensor_12",
    "sensor_13", "sensor_14", "sensor_15", "sensor_17", "sensor_20", "sensor_21"
]

COLUMNS = [
    "unit_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"
] + [f"sensor_{i}" for i in range(1, 22)]


class UAVTurbofanLSTM(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
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


class RULPredictor:
    def __init__(self, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = joblib.load(scaler_path)
        self.model = UAVTurbofanLSTM(input_dim=len(FEATURE_COLS)).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()

    def classify_tactical_status(self, predicted_rul: float) -> str:
        if predicted_rul <= 30:
            return "KRİTİK (Acil Üsse Dönüş / Bakım)"
        elif predicted_rul <= 65:
            return "UYARI (Planlı Bakım Kuyruğuna Al)"
        return "NORMAL (Göreve Devam)"

    def predict_from_window(self, window_df: pd.DataFrame) -> dict:
        if len(window_df) < SEQUENCE_LENGTH:
            raise ValueError(f"Tahmin için en az {SEQUENCE_LENGTH} çevrim verisi gereklidir.")

        recent_window = window_df.tail(SEQUENCE_LENGTH)[FEATURE_COLS]
        scaled_data = self.scaler.transform(recent_window)
        input_tensor = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred_rul = float(self.model(input_tensor).item())

        pred_rul = max(0.0, round(pred_rul, 2))
        return {
            "predicted_rul": pred_rul,
            "status": self.classify_tactical_status(pred_rul)
        }


if __name__ == "__main__":
    predictor = RULPredictor()
    df = pd.read_csv(RAW_DATA_PATH, sep=r"\s+", header=None, names=COLUMNS)

    # UAV-01 (unit_id=1, toplam 192 çevrim) için erken, orta ve geç dönem testi
    uav_1 = df[df["unit_id"] == 1].sort_values("cycle")
    max_cycle = uav_1["cycle"].max()

    print("--- UAV-BAYRAKTAR-01 Kestirimci Bakım Doğrulama Testi ---")
    for end_cycle in [40, 120, 185]:
        window = uav_1[uav_1["cycle"] <= end_cycle]
        actual_rul = max_cycle - end_cycle
        result = predictor.predict_from_window(window)
        print(
            f"Çevrim {end_cycle:03d}/{max_cycle} | Gerçek RUL: {actual_rul:03d} | "
            f"Tahmin RUL: {result['predicted_rul']:06.2f} | Durum: {result['status']}"
        )