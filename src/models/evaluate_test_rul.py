import json
import os
import urllib.request
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, precision_score, recall_score

from predict_rul import COLUMNS, FEATURE_COLS, MODEL_PATH, SCALER_PATH, SEQUENCE_LENGTH, UAVTurbofanLSTM

RAW_DIR = os.path.join("data", "raw")
TEST_DATA_PATH = os.path.join(RAW_DIR, "test_FD001.txt")
RUL_GT_PATH = os.path.join(RAW_DIR, "RUL_FD001.txt")
REPORT_OUT_PATH = os.path.join("data", "processed", "ip7_test_results.json")

BASE_URL = "https://raw.githubusercontent.com/hankroark/Turbofan-Engine-Degradation/master/CMAPSSData"


def ensure_test_files():
    os.makedirs(RAW_DIR, exist_ok=True)
    for fname, fpath in [("test_FD001.txt", TEST_DATA_PATH), ("RUL_FD001.txt", RUL_GT_PATH)]:
        if not os.path.exists(fpath):
            url = f"{BASE_URL}/{fname}"
            print(f"İndiriliyor: {fname} ...")
            urllib.request.urlretrieve(url, fpath)


def nasa_phm08_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """NASA PHM08 asimetrik skor fonksiyonu (geç tahmin daha ağır cezalandırılır)."""
    diff = y_pred - y_true
    scores = np.where(diff < 0, np.exp(-diff / 13.0) - 1.0, np.exp(diff / 10.0) - 1.0)
    return float(np.sum(scores))


def evaluate_on_test_set():
    ensure_test_files()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = joblib.load(SCALER_PATH)
    model = UAVTurbofanLSTM(input_dim=len(FEATURE_COLS)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    test_df = pd.read_csv(TEST_DATA_PATH, sep=r"\s+", header=None, names=COLUMNS)
    y_true_raw = pd.read_csv(RUL_GT_PATH, header=None)[0].values
    y_true_clipped = np.clip(y_true_raw, a_min=0, a_max=125)

    # Her bir test motorunun (unit_id 1..100) son 30 çevrimlik penceresini hazırla
    test_sequences = []
    for unit_id in sorted(test_df["unit_id"].unique()):
        unit_data = test_df[test_df["unit_id"] == unit_id].sort_values("cycle")[FEATURE_COLS]
        scaled = scaler.transform(unit_data)
        if len(scaled) >= SEQUENCE_LENGTH:
            seq = scaled[-SEQUENCE_LENGTH:]
        else:
            pad = np.repeat(scaled[0:1], SEQUENCE_LENGTH - len(scaled), axis=0)
            seq = np.vstack([pad, scaled])
        test_sequences.append(seq)

    X_test = torch.tensor(np.array(test_sequences), dtype=torch.float32).to(device)
    with torch.no_grad():
        y_pred = model(X_test).squeeze(1).cpu().numpy()
    y_pred = np.clip(y_pred, a_min=0, a_max=125)

    # Regresyon Metrikleri
    rmse_clipped = float(np.sqrt(mean_squared_error(y_true_clipped, y_pred)))
    mae_clipped = float(mean_absolute_error(y_true_clipped, y_pred))
    phm_score = nasa_phm08_score(y_true_clipped, y_pred)

    # Erken Arıza Tespiti Sınıflandırma Metrikleri (Kritik: RUL <= 30, Planlı Bakım: RUL <= 50)
    crit_true = (y_true_clipped <= 30).astype(int)
    crit_pred = (y_pred <= 30).astype(int)
    f1_crit = float(f1_score(crit_true, crit_pred))
    prec_crit = float(precision_score(crit_true, crit_pred, zero_division=0))
    rec_crit = float(recall_score(crit_true, crit_pred, zero_division=0))

    warn_true = (y_true_clipped <= 50).astype(int)
    warn_pred = (y_pred <= 50).astype(int)
    f1_warn = float(f1_score(warn_true, warn_pred))

    print("\n=====================================================================")
    print("  İP7: NASA C-MAPSS test_FD001 KÖR TEST DEĞERLENDİRME SONUÇLARI")
    print("=====================================================================")
    print(f"Test Motor Sayısı           : {len(y_true_clipped)} ünite")
    print(f"Test RMSE (Hedef <= 15.0)   : {rmse_clipped:.2f} çevrim  --> {'BAŞARILI' if rmse_clipped <= 15 else 'GELİŞTİRİLMELİ'}")
    print(f"Test MAE                    : {mae_clipped:.2f} çevrim")
    print(f"NASA PHM08 Asimetrik Skor   : {phm_score:.2f}")
    print("---------------------------------------------------------------------")
    print(f"Kritik Arıza Eşiği (RUL<=30): Precision={prec_crit:.2%} | Recall={rec_crit:.2%} | F1={f1_crit:.2%}")
    print(f"Bakım Penceresi    (RUL<=50): F1-Score={f1_warn:.2%} (Hedef >= %80)")
    print("=====================================================================\n")

    results = {
        "dataset": "NASA C-MAPSS test_FD001",
        "num_test_units": int(len(y_true_clipped)),
        "test_rmse": round(rmse_clipped, 4),
        "test_mae": round(mae_clipped, 4),
        "nasa_phm08_score": round(phm_score, 2),
        "critical_threshold_30": {
            "precision": round(prec_crit, 4),
            "recall": round(rec_crit, 4),
            "f1_score": round(f1_crit, 4)
        },
        "maintenance_threshold_50": {
            "f1_score": round(f1_warn, 4)
        }
    }

    os.makedirs(os.path.dirname(REPORT_OUT_PATH), exist_ok=True)
    with open(REPORT_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Rapor metrikleri kaydedildi -> {REPORT_OUT_PATH}")


if __name__ == "__main__":
    evaluate_on_test_set()