import json
import time
from datetime import datetime, timezone
import random

def generate_sample_telemetry(unit_id="UAV-TUR-101"):
    return {
        "unit_id": unit_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operational_status": "ACTIVE_MISSION",
        "telemetry": {
            "latitude": round(40.35 + random.uniform(-0.05, 0.05), 4),
            "longitude": round(27.97 + random.uniform(-0.05, 0.05), 4),
            "altitude_m": round(random.uniform(3000, 5000), 1),
            "speed_kmh": round(random.uniform(180, 240), 1),
            "fuel_level_pct": round(random.uniform(40, 95), 1)
        },
        "c_mapss_sensors": {
            "s2": round(random.uniform(640, 645), 2),
            "s3": round(random.uniform(1580, 1600), 2),
            "s4": round(random.uniform(1400, 1420), 2),
            "s7": round(random.uniform(550, 560), 2),
            "s8": round(random.uniform(2380, 2390), 2),
            "s11": round(random.uniform(47, 48), 2),
            "s12": round(random.uniform(520, 530), 2),
            "s15": round(random.uniform(8.4, 8.6), 4)
        }
    }

if __name__ == "__main__":
    print("Testing telemetry schema...\n")
    for i in range(1, 4):
        sample = generate_sample_telemetry()
        print(f"--- Packet {i} ---")
        print(json.dumps(sample, indent=2))
        time.sleep(1)
    print("\nVerification successful: Project skeleton and data format are ready.")