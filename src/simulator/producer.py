import json
import os
import sys
import time
from datetime import datetime, timezone
import pandas as pd
from kafka import KafkaProducer

BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC_NAME = "telemetry.tactical.raw"
DATA_PATH = os.path.join("data", "raw", "train_FD001.txt")

COLUMNS = [
    "unit_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"
] + [f"sensor_{i}" for i in range(1, 22)]

def create_producer():
    try:
        return KafkaProducer(
            bootstrap_servers=BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            acks=1,
            retries=3
        )
    except Exception as e:
        print(f"Kafka Producer baslatilamadi: {e}", file=sys.stderr)
        sys.exit(1)

def run_tactical_stream(unit_filter=1, stream_interval_sec=0.5):
    if not os.path.exists(DATA_PATH):
        print(f"Hata: {DATA_PATH} bulunamadi. Once download_cmapss.py dosyasini calistirin.")
        sys.exit(1)

    print(f"Veri kaynagi okunuyor: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, sep=r"\s+", header=None, names=COLUMNS)
    
    # Motor 1 (UAV-01) ucus dongusunu filtrele
    unit_data = df[df["unit_id"] == unit_filter].sort_values("cycle")
    total_cycles = len(unit_data)
    print(f"Motor {unit_filter} filtrelendi. Toplam cevrim (cycle) sayisi: {total_cycles}")

    producer = create_producer()
    print(f"Kafka yayini baslatiliyor -> Topic: '{TOPIC_NAME}' | Frekans: {stream_interval_sec} sn/mesaj\n")

    try:
        for _, row in unit_data.iterrows():
            current_cycle = int(row["cycle"])
            payload = {
                "uav_id": f"UAV-BAYRAKTAR-{int(row['unit_id']):02d}",
                "subsystem": "TURBOFAN_ENGINE",
                "cycle": current_cycle,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "operational_settings": {
                    "setting_1": float(row["op_setting_1"]),
                    "setting_2": float(row["op_setting_2"]),
                    "setting_3": float(row["op_setting_3"])
                },
                "sensor_measurements": {
                    f"s_{i}": float(row[f"sensor_{i}"]) for i in range(1, 22)
                }
            }
            
            producer.send(TOPIC_NAME, value=payload)
            print(f"-> [Cycle {current_cycle:03d}/{total_cycles}] UAV-BAYRAKTAR-{int(row['unit_id']):02d} telemetrisi gonderildi.")
            time.sleep(stream_interval_sec)

        producer.flush()
        print("\nVeri akisi basariyla tamamlandi.")
    except KeyboardInterrupt:
        print("\nVeri akisi kullanici tarafindan durduruldu (Ctrl+C).")
    finally:
        producer.close()

if __name__ == "__main__":
    # Ilk motorun tum yasam dongusunu 0.5 saniyede bir gonderir
    run_tactical_stream(unit_filter=1, stream_interval_sec=0.5)