import json
import os
import sys
import time
import pandas as pd
from kafka import KafkaProducer
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.models.predict_rul import COLUMNS, FEATURE_COLS, RAW_DATA_PATH

BOOTSTRAP_SERVERS = "localhost:9092"
LIVE_TOPIC = "telemetry.tactical.live"

# 5 İHA için farklı aşınma başlangıç noktaları (30 çevrim penceresi dolup ilerleyecek şekilde)
FLEET_CONFIG = [
    {"uav_id": "UAV-BAYRAKTAR-01", "unit_id": 1, "start_cycle": 150, "end_cycle": 192, "lat": 39.85, "lon": 26.60},
    {"uav_id": "UAV-BAYRAKTAR-02", "unit_id": 2, "start_cycle": 205, "end_cycle": 247, "lat": 39.10, "lon": 26.85},
    {"uav_id": "UAV-BAYRAKTAR-03", "unit_id": 3, "start_cycle": 100, "end_cycle": 142, "lat": 38.40, "lon": 26.95},
    {"uav_id": "UAV-BAYRAKTAR-04", "unit_id": 4, "start_cycle": 15,  "end_cycle": 57,  "lat": 40.15, "lon": 27.40},
    {"uav_id": "UAV-BAYRAKTAR-05", "unit_id": 5, "start_cycle": 180, "end_cycle": 222, "lat": 37.20, "lon": 28.10},
]


def ensure_topic():
    admin = KafkaAdminClient(bootstrap_servers=BOOTSTRAP_SERVERS)
    try:
        admin.create_topics([NewTopic(name=LIVE_TOPIC, num_partitions=1, replication_factor=1)])
    except TopicAlreadyExistsError:
        pass
    finally:
        admin.close()


def stream_live_fleet():
    ensure_topic()
    df = pd.read_csv(RAW_DATA_PATH, sep=r"\s+", header=None, names=COLUMNS)

    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks=1
    )

    print("=== 5 İHA CANLI TAKTİK TELEMETRİ AKIŞI BAŞLATILIYOR ===")
    total_steps = 42  # İlk 30 adım pencereyi doldurur, sonraki 12 adım canlı RUL düşüşünü gösterir

    for step in range(total_steps):
        for cfg in FLEET_CONFIG:
            target_cycle = cfg["start_cycle"] + step
            row_df = df[(df["unit_id"] == cfg["unit_id"]) & (df["cycle"] == target_cycle)]
            if row_df.empty:
                continue

            row = row_df.iloc[0]
            packet = {
                "uav_id": cfg["uav_id"],
                "unit_id": cfg["unit_id"],
                "cycle": int(target_cycle),
                "lat": round(cfg["lat"] + (step * 0.002), 5),
                "lon": round(cfg["lon"] + (step * 0.0025), 5),
                "sent_ts_ms": time.time() * 1000.0
            }
            for feat in FEATURE_COLS:
                packet[feat] = float(row[feat])

            producer.send(LIVE_TOPIC, value=packet)

        producer.flush()
        print(f"Adım {step + 1:02d}/{total_steps:02d} -> 5 İHA telemetri paketi Kafka'ya iletildi.")
        time.sleep(0.5)

    producer.close()
    print("=== CANLI TELEMETRİ AKIŞI TAMAMLANDI ===")


if __name__ == "__main__":
    stream_live_fleet()