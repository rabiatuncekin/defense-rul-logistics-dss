from collections import deque
from datetime import datetime, timezone
import json
import os
import shutil
import sqlite3
import sys
import time
import urllib.request
import joblib
import numpy as np
import pandas as pd
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.models.predict_rul import FEATURE_COLS, MODEL_PATH, SCALER_PATH, SEQUENCE_LENGTH, UAVTurbofanLSTM

BOOTSTRAP_SERVERS = "localhost:9092"
LIVE_TOPIC = "telemetry.tactical.live"
DB_PATH = os.path.join("data", "processed", "tactical_dss.db")
IP9_REPORT_PATH = os.path.join("data", "processed", "ip9_inference_benchmark.json")
CHECKPOINT_DIR = os.path.abspath(os.path.join("data", "processed", "checkpoints", "ip9_stream"))

uav_windows: dict[str, deque] = {}
inference_latencies_ms: list[float] = []


def setup_windows_hadoop():
    """Windows üzerinde PySpark Structured Streaming için gerekli winutils.exe ve hadoop.dll dosyalarını hazırlar."""
    if os.name != "nt":
        return

    hadoop_home = os.path.abspath(".hadoop")
    bin_dir = os.path.join(hadoop_home, "bin")
    os.makedirs(bin_dir, exist_ok=True)

    binaries = {
        "winutils.exe": [
            "https://raw.githubusercontent.com/cdarlint/winutils/master/hadoop-3.3.5/bin/winutils.exe",
            "https://raw.githubusercontent.com/kontext-tech/winutils/master/hadoop-3.3.5/bin/winutils.exe",
        ],
        "hadoop.dll": [
            "https://raw.githubusercontent.com/cdarlint/winutils/master/hadoop-3.3.5/bin/hadoop.dll",
            "https://raw.githubusercontent.com/kontext-tech/winutils/master/hadoop-3.3.5/bin/hadoop.dll",
        ],
    }

    for fname, urls in binaries.items():
        dest = os.path.join(bin_dir, fname)
        if not os.path.exists(dest) or os.path.getsize(dest) == 0:
            for url in urls:
                try:
                    print(f"Windows Hadoop bileşeni indiriliyor: {fname} ...")
                    urllib.request.urlretrieve(url, dest)
                    if os.path.exists(dest) and os.path.getsize(dest) > 0:
                        break
                except Exception:
                    continue

    os.environ["HADOOP_HOME"] = hadoop_home
    os.environ["hadoop.home.dir"] = hadoop_home
    os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")


def build_spark_session() -> SparkSession:
    spark_ver = pyspark.__version__
    scala_ver = "2.13" if spark_ver.startswith("4.") else "2.12"
    kafka_pkg = f"org.apache.spark:spark-sql-kafka-0-10_{scala_ver}:{spark_ver}"

    spark = (
        SparkSession.builder
        .appName("TacticalUAV-Live-LSTM-Inference")
        .master("local[*]")
        .config("spark.jars.packages", kafka_pkg)
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def get_telemetry_schema() -> StructType:
    fields = [
        StructField("uav_id", StringType(), True),
        StructField("unit_id", IntegerType(), True),
        StructField("cycle", IntegerType(), True),
        StructField("lat", DoubleType(), True),
        StructField("lon", DoubleType(), True),
        StructField("sent_ts_ms", DoubleType(), True),
    ]
    for feat in FEATURE_COLS:
        fields.append(StructField(feat, DoubleType(), True))
    return StructType(fields)


def classify_risk(rul: float) -> str:
    if rul <= 30.0:
        return "KRİTİK (Acil Dönüş)"
    elif rul <= 65.0:
        return "UYARI (Planlı Bakım)"
    return "NORMAL (Görevde)"


def persist_live_predictions(records: list[dict]):
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    df_out = pd.DataFrame(records)
    df_out.to_sql("live_stream_predictions", conn, if_exists="replace", index=False)
    conn.close()


def main():
    setup_windows_hadoop()

    if os.path.exists(CHECKPOINT_DIR):
        shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = joblib.load(SCALER_PATH)
    model = UAVTurbofanLSTM(input_dim=len(FEATURE_COLS)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    spark = build_spark_session()
    schema = get_telemetry_schema()

    raw_stream = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", BOOTSTRAP_SERVERS)
        .option("subscribe", LIVE_TOPIC)
        .option("startingOffsets", "latest")
        .load()
    )

    parsed_stream = (
        raw_stream
        .select(from_json(col("value").cast("string"), schema).alias("data"))
        .select("data.*")
    )

    def process_micro_batch(batch_df, batch_id: int):
        if batch_df.isEmpty():
            return

        t_batch_start = time.perf_counter()
        pdf = batch_df.toPandas().sort_values(["uav_id", "cycle"])

        for _, row in pdf.iterrows():
            uid = row["uav_id"]
            if uid not in uav_windows:
                uav_windows[uid] = deque(maxlen=SEQUENCE_LENGTH)
            uav_windows[uid].append({
                "cycle": int(row["cycle"]),
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "features": [float(row[c]) for c in FEATURE_COLS]
            })

        ready_uavs = []
        batch_sequences = []
        for uid, buf in sorted(uav_windows.items()):
            if len(buf) == SEQUENCE_LENGTH:
                seq_raw = np.array([item["features"] for item in buf], dtype=np.float32)
                seq_scaled = scaler.transform(pd.DataFrame(seq_raw, columns=FEATURE_COLS))
                batch_sequences.append(seq_scaled)
                ready_uavs.append((uid, buf[-1]))

        if not ready_uavs:
            print(f"[Batch #{batch_id:02d}] Tampon dolduruluyor... (Mevcut pencere: {max(len(b) for b in uav_windows.values())}/{SEQUENCE_LENGTH})")
            return

        t_infer_start = time.perf_counter()
        input_tensor = torch.tensor(np.array(batch_sequences), dtype=torch.float32).to(device)
        with torch.no_grad():
            preds = model(input_tensor).squeeze(-1).cpu().numpy()
        if preds.ndim == 0:
            preds = np.array([float(preds)])

        infer_ms = (time.perf_counter() - t_infer_start) * 1000.0
        total_batch_ms = (time.perf_counter() - t_batch_start) * 1000.0
        inference_latencies_ms.append(infer_ms)

        live_records = []
        print(f"\n--- [PySpark Micro-Batch #{batch_id:02d}] LSTM Inference: {infer_ms:.2f} ms | Toplam: {total_batch_ms:.2f} ms (Hedef < 500 ms) ---")
        for (uid, latest), raw_rul in zip(ready_uavs, preds):
            rul_val = round(max(0.0, float(raw_rul)), 2)
            status = classify_risk(rul_val)
            print(f"  {uid:<18} | Çevrim: {latest['cycle']:03d} | Canlı RUL: {rul_val:06.2f} | Durum: {status}")
            live_records.append({
                "uav_id": uid,
                "cycle": latest["cycle"],
                "lat": latest["lat"],
                "lon": latest["lon"],
                "predicted_rul": rul_val,
                "status": status,
                "inference_ms": round(infer_ms, 2),
                "updated_at": datetime.now(timezone.utc).isoformat()
            })

        persist_live_predictions(live_records)

        os.makedirs(os.path.dirname(IP9_REPORT_PATH), exist_ok=True)
        with open(IP9_REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "batches_evaluated": len(inference_latencies_ms),
                "avg_lstm_inference_ms": round(float(np.mean(inference_latencies_ms)), 2),
                "max_lstm_inference_ms": round(float(np.max(inference_latencies_ms)), 2),
                "target_threshold_ms": 500.0,
                "status": "PASSED" if np.max(inference_latencies_ms) < 500.0 else "FAILED"
            }, f, indent=2)

    print("PySpark Structured Streaming + PyTorch LSTM Çıkarım Hattı Dinlemede...")
    query = (
        parsed_stream.writeStream
        .option("checkpointLocation", CHECKPOINT_DIR)
        .foreachBatch(process_micro_batch)
        .trigger(processingTime="1 second")
        .start()
    )
    query.awaitTermination()


if __name__ == "__main__":
    main()