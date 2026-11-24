import json
import threading
import os
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from kafka import KafkaConsumer, KafkaProducer
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError

BOOTSTRAP_SERVERS = "localhost:9092"
BENCHMARK_TOPIC = "telemetry.tactical.benchmark"
RAW_DATA_PATH = os.path.join("data", "raw", "train_FD001.txt")
BENCHMARK_OUT_PATH = os.path.join("data", "processed", "ip5_benchmark_results.json")

COLUMNS = [
    "unit_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"
] + [f"sensor_{i}" for i in range(1, 22)]

FLEET_PROFILES = [
    {"uav_id": "UAV-BAYRAKTAR-01", "unit_id": 1, "base_lat": 39.85, "base_lon": 26.60, "speed_kmh": 130.0},
    {"uav_id": "UAV-BAYRAKTAR-02", "unit_id": 2, "base_lat": 39.10, "base_lon": 26.85, "speed_kmh": 135.0},
    {"uav_id": "UAV-BAYRAKTAR-03", "unit_id": 3, "base_lat": 38.40, "base_lon": 26.95, "speed_kmh": 140.0},
    {"uav_id": "UAV-BAYRAKTAR-04", "unit_id": 4, "base_lat": 40.15, "base_lon": 27.40, "speed_kmh": 140.0},
    {"uav_id": "UAV-BAYRAKTAR-05", "unit_id": 5, "base_lat": 37.20, "base_lon": 28.10, "speed_kmh": 130.0},
]


def prepare_topic():
    admin = KafkaAdminClient(bootstrap_servers=BOOTSTRAP_SERVERS)
    try:
        admin.create_topics([NewTopic(name=BENCHMARK_TOPIC, num_partitions=1, replication_factor=1)])
    except TopicAlreadyExistsError:
        pass
    finally:
        admin.close()


def build_tactical_packets(df: pd.DataFrame, target_count: int = 5000) -> list[bytes]:
    """C-MAPSS sensör verilerini taktik GPS, hız, yakıt ve görev durumuyla birleştirerek JSON paketleri hazırlar."""
    packets = []
    subset = df[df["unit_id"].isin([1, 2, 3, 4, 5])].reset_index(drop=True)
    total_rows = len(subset)

    for i in range(target_count):
        row = subset.iloc[i % total_rows]
        profile = FLEET_PROFILES[int(row["unit_id"]) - 1]
        cycle = int(row["cycle"])

        payload = {
            "uav_id": profile["uav_id"],
            "unit_id": int(row["unit_id"]),
            "cycle": cycle,
            "gps": {
                "lat": round(profile["base_lat"] + (cycle * 0.0012), 5),
                "lon": round(profile["base_lon"] + (cycle * 0.0015), 5),
                "altitude_m": round(5200.0 + (i % 150), 1),
            },
            "tactical_state": {
                "cruise_speed_kmh": profile["speed_kmh"],
                "fuel_pct": round(max(12.0, 98.0 - (cycle * 0.35)), 1),
                "mission_mode": "ISR_PATROL" if cycle < 150 else "RTB_ALERT",
            },
            "sensors": {f"sensor_{s}": float(row[f"sensor_{s}"]) for s in range(1, 22)},
            "sent_ts_ms": 0.0,  # Gönderim anında doldurulacak
        }
        packets.append(payload)
    return packets


def run_throughput_and_latency_benchmark(num_messages: int = 3000):
    prepare_topic()
    df = pd.read_csv(RAW_DATA_PATH, sep=r"\s+", header=None, names=COLUMNS)
    packets = build_tactical_packets(df, target_count=num_messages)

    consumer = KafkaConsumer(
        BENCHMARK_TOPIC,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        auto_offset_reset="latest",
        consumer_timeout_ms=6000,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    )
    consumer.poll(timeout_ms=500)

    latencies_ms = []
    consumer_ready = threading.Event()

    def consume_worker():
        consumer_ready.set()
        for msg in consumer:
            recv_ts_ms = time.time() * 1000.0
            sent_ts_ms = msg.value["sent_ts_ms"]
            latencies_ms.append(max(0.2, recv_ts_ms - sent_ts_ms))
            if len(latencies_ms) >= num_messages:
                break

    t_consumer = threading.Thread(target=consume_worker, daemon=True)
    t_consumer.start()
    consumer_ready.wait()

    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        linger_ms=2,
        batch_size=16384,
        acks=1,
    )

    print(f"--- İP4 & İP5 EŞZAMANLI TAKTİK TELEMETRİ AKIŞ TESTİ ({num_messages} PAKET) ---")
    t_start = time.perf_counter()

    for idx, pkt in enumerate(packets, start=1):
        pkt["sent_ts_ms"] = time.time() * 1000.0
        producer.send(BENCHMARK_TOPIC, value=json.dumps(pkt).encode("utf-8"))
        # ~1.350 mesaj/sn taktik akış hızını simüle et
        if idx % 150 == 0:
            producer.flush()
            time.sleep(0.08)

    producer.flush()
    t_produce_end = time.perf_counter()
    t_consumer.join(timeout=10.0)
    t_end = time.perf_counter()

    received = len(latencies_ms)
    producer.close()
    consumer.close()

    produce_rate = num_messages / (t_produce_end - t_start)
    e2e_rate = received / (t_end - t_start)
    avg_latency = float(np.mean(latencies_ms))
    p95_latency = float(np.percentile(latencies_ms, 95))
    loss_rate = ((num_messages - received) / num_messages) * 100.0

    print(f"Gönderilen / Alınan Paket   : {num_messages} / {received} (Kayıp: %{loss_rate:.2f})")
    print(f"Üretici Akış Hızı           : {produce_rate:,.1f} mesaj/sn")
    print(f"Uçtan Uca İşleme Hızı       : {e2e_rate:,.1f} mesaj/sn (Hedef >= 1.000 mesaj/sn)")
    print(f"Ortalama E2E Gecikme        : {avg_latency:.2f} ms (Hedef < 500 ms)")
    print(f"P95 Kuyruk Gecikmesi        : {p95_latency:.2f} ms")

    results = {
        "total_messages": num_messages,
        "received_messages": received,
        "packet_loss_pct": round(loss_rate, 4),
        "producer_throughput_msg_per_sec": round(produce_rate, 2),
        "e2e_throughput_msg_per_sec": round(e2e_rate, 2),
        "avg_e2e_latency_ms": round(avg_latency, 2),
        "p95_e2e_latency_ms": round(p95_latency, 2),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    os.makedirs(os.path.dirname(BENCHMARK_OUT_PATH), exist_ok=True)
    with open(BENCHMARK_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nBenchmark sonuçları kaydedildi -> {BENCHMARK_OUT_PATH}")


if __name__ == "__main__":
    run_throughput_and_latency_benchmark(num_messages=5000)

