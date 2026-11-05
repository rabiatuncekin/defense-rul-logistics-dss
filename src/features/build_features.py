import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

RAW_DATA_PATH = os.path.join("data", "raw", "train_FD001.txt")
PROCESSED_DATA_DIR = os.path.join("data", "processed")
SCALER_PATH = os.path.join(PROCESSED_DATA_DIR, "scaler.pkl")
TRAIN_PROCESSED_PATH = os.path.join(PROCESSED_DATA_DIR, "train_FD001_processed.parquet")

COLUMNS = [
    "unit_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"
] + [f"sensor_{i}" for i in range(1, 22)]

# RUL kestiriminde anlamlı varyans gösteren 14 kritik sensör
RELEVANT_SENSORS = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8", "sensor_9",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_17", "sensor_20", "sensor_21"
]
FEATURE_COLS = ["op_setting_1", "op_setting_2"] + RELEVANT_SENSORS

def process_and_scale_data():
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    print(f"Ham veri okunuyor: {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH, sep=r"\s+", header=None, names=COLUMNS)

    # 1. Her bir motor için gerçek RUL (Kalan Faydalı Ömür) etiketlerini hesapla
    max_cycle = df.groupby("unit_id")["cycle"].max().reset_index()
    max_cycle.columns = ["unit_id", "max_cycle"]
    df = df.merge(max_cycle, on="unit_id", how="left")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df.drop(columns=["max_cycle"], inplace=True)

    # 2. Min-Max Ölçekleyiciyi 16 öznitelik (2 ayar + 14 sensör) üzerinde eğit
    print("Sensör öznitelikleri [0, 1] aralığına normalize ediliyor...")
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df[FEATURE_COLS])
    
    scaled_df = pd.DataFrame(scaled_features, columns=FEATURE_COLS)
    scaled_df["unit_id"] = df["unit_id"]
    scaled_df["cycle"] = df["cycle"]
    scaled_df["RUL"] = df["RUL"]

    # 3. Scaler nesnesini ve işlenmiş veri kümesini kaydet
    joblib.dump(scaler, SCALER_PATH)
    scaled_df.to_parquet(TRAIN_PROCESSED_PATH, index=False)
    
    print(f"\nScaler parametreleri kaydedildi: {SCALER_PATH}")
    print(f"İşlenmiş veri kümesi kaydedildi: {TRAIN_PROCESSED_PATH}")
    print(f"Kayıt boyutu: {scaled_df.shape[0]} satır, {scaled_df.shape[1]} sütun.")

if __name__ == "__main__":
    process_and_scale_data()