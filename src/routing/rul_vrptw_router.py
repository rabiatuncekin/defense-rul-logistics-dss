import json
import math
import os
import sys
import networkx as nx
import pandas as pd

# Proje kök dizinini import yoluna ekle
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.models.predict_rul import RULPredictor, RAW_DATA_PATH, COLUMNS

ROUTING_OUTPUT_PATH = os.path.join("data", "processed", "tactical_routing_plan.json")


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """İki koordinat arasındaki kuş uçuşu mesafeyi (km) hesaplar."""
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class TacticalRULRouter:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.predictor = RULPredictor()
        self._build_tactical_network()

    def _build_tactical_network(self):
        """Ana Harekat Üsleri (MOB) ve İleri Bakım Depolarını (FMD) ağa ekler."""
        depots = [
            {
                "node_id": "MOB-BANDIRMA",
                "name": "Bandırma Ana Harekat ve Ağır Bakım Üssü",
                "type": "DEPOT",
                "lat": 40.3178,
                "lon": 27.9775,
                "capacity": 3,
                "tw_start_min": 0,
                "tw_end_min": 180,
                "service_level": "FULL_OVERHAUL"
            },
            {
                "node_id": "FMD-EDREMIT",
                "name": "Edremit İleri Bakım ve İkmal Noktası",
                "type": "DEPOT",
                "lat": 39.5546,
                "lon": 27.0138,
                "capacity": 2,
                "tw_start_min": 10,
                "tw_end_min": 120,
                "service_level": "RAPID_MAINTENANCE"
            },
            {
                "node_id": "FMD-DALAMAN",
                "name": "Dalaman Taktik Destek Üssü",
                "type": "DEPOT",
                "lat": 36.7131,
                "lon": 28.7925,
                "capacity": 2,
                "tw_start_min": 15,
                "tw_end_min": 150,
                "service_level": "RAPID_MAINTENANCE"
            }
        ]

        for depot in depots:
            self.graph.add_node(depot["node_id"], **depot)

    def register_fleet_telemetry(self, df: pd.DataFrame) -> list:
        """5 farklı İHA'nın güncel uçuş çevrimlerinden RUL kestirimi yapar ve çizgeye ekler."""
        fleet_scenarios = [
            {"unit_id": 1, "uav_id": "UAV-BAYRAKTAR-01", "cycle": 185, "lat": 39.85, "lon": 26.60, "cruise_kmh": 130},
            {"unit_id": 2, "uav_id": "UAV-BAYRAKTAR-02", "cycle": 240, "lat": 39.10, "lon": 26.85, "cruise_kmh": 135},
            {"unit_id": 3, "uav_id": "UAV-BAYRAKTAR-03", "cycle": 135, "lat": 38.40, "lon": 26.95, "cruise_kmh": 140},
            {"unit_id": 4, "uav_id": "UAV-BAYRAKTAR-04", "cycle": 50,  "lat": 40.15, "lon": 27.40, "cruise_kmh": 140},
            {"unit_id": 5, "uav_id": "UAV-BAYRAKTAR-05", "cycle": 215, "lat": 37.20, "lon": 28.10, "cruise_kmh": 130},
        ]

        fleet_states = []
        for item in fleet_scenarios:
            unit_df = df[(df["unit_id"] == item["unit_id"]) & (df["cycle"] <= item["cycle"])].sort_values("cycle")
            pred = self.predictor.predict_from_window(unit_df)

            uav_node = {
                "node_id": item["uav_id"],
                "uav_id": item["uav_id"],
                "type": "UAV",
                "unit_id": item["unit_id"],
                "current_cycle": item["cycle"],
                "lat": item["lat"],
                "lon": item["lon"],
                "cruise_kmh": item["cruise_kmh"],
                "predicted_rul": pred["predicted_rul"],
                "status": pred["status"]
            }
            self.graph.add_node(item["uav_id"], **uav_node)
            fleet_states.append(uav_node)

            # İHA ile tüm bakım üsleri arasındaki uçuş maliyet kenarlarını oluştur
            for node_id, data in self.graph.nodes(data=True):
                if data.get("type") == "DEPOT":
                    dist_km = haversine_km(item["lat"], item["lon"], data["lat"], data["lon"])
                    flight_min = (dist_km / item["cruise_kmh"]) * 60.0
                    # 1 uçuş çevrimi yaklaşık 15 dakikalık sorti yüküne eşdeğer modellenir
                    required_cycles = round(flight_min / 15.0, 2)
                    self.graph.add_edge(
                        item["uav_id"],
                        node_id,
                        distance_km=round(dist_km, 2),
                        flight_time_min=round(flight_min, 1),
                        required_cycles=required_cycles
                    )

        return fleet_states

    def solve_rul_vrptw(self, fleet_states: list) -> list:
        """
        RUL-VRPTW Çözücü:
        Kalan ömrü kritik/uyarı seviyesindeki İHA'ları en düşük RUL önceliğiyle,
        sağkalım kısıtı (required_cycles + güvenlik payı <= predicted_rul) ve
        zaman penceresi altında en uygun bakım üssüne yönlendirir.
        """
        # Aciliyet sırasına göre en düşük RUL değerine sahip İHA'dan başla
        sorted_uavs = sorted(fleet_states, key=lambda x: x["predicted_rul"])
        remaining_capacity = {
            n: d["capacity"] for n, d in self.graph.nodes(data=True) if d.get("type") == "DEPOT"
        }

        routing_decisions = []
        safety_margin_cycles = 2.0

        for uav in sorted_uavs:
            uav_id = uav["uav_id"]
            rul = uav["predicted_rul"]

            if rul > 65:
                routing_decisions.append({
                    "uav_id": uav_id,
                    "predicted_rul": rul,
                    "status": uav["status"],
                    "action": "GÖREVDE KAL",
                    "assigned_depot": "-",
                    "distance_km": 0.0,
                    "eta_min": 0.0,
                    "feasible": True
                })
                continue

            # Uygun üs adaylarını değerlendir (Kapasite + Zaman Penceresi + RUL Sağkalım Kısıtı)
            feasible_candidates = []
            for depot_id in self.graph.successors(uav_id):
                depot = self.graph.nodes[depot_id]
                edge = self.graph.edges[uav_id, depot_id]

                if remaining_capacity[depot_id] <= 0:
                    continue

                eta = edge["flight_time_min"]
                in_time_window = depot["tw_start_min"] <= eta <= depot["tw_end_min"]
                survives_flight = (edge["required_cycles"] + safety_margin_cycles) <= rul

                if in_time_window and survives_flight:
                    # Maliyet fonksiyonu: Uçuş süresi + RUL risk ağırlığı
                    cost = eta * (1.5 if rul <= 30 else 1.0)
                    feasible_candidates.append((cost, depot_id, edge))

            if feasible_candidates:
                feasible_candidates.sort(key=lambda x: x[0])
                _, best_depot, best_edge = feasible_candidates[0]
                remaining_capacity[best_depot] -= 1

                routing_decisions.append({
                    "uav_id": uav_id,
                    "predicted_rul": rul,
                    "status": uav["status"],
                    "action": "BAKIM ROTASI ATANDI",
                    "assigned_depot": best_depot,
                    "distance_km": best_edge["distance_km"],
                    "eta_min": best_edge["flight_time_min"],
                    "feasible": True
                })
            else:
                routing_decisions.append({
                    "uav_id": uav_id,
                    "predicted_rul": rul,
                    "status": uav["status"],
                    "action": "ACİL İNİŞ / MOBİL EKİP SEVKİ",
                    "assigned_depot": "EN YAKIN ACİL PİST",
                    "distance_km": 0.0,
                    "eta_min": 0.0,
                    "feasible": False
                })

        return routing_decisions


if __name__ == "__main__":
    df = pd.read_csv(RAW_DATA_PATH, sep=r"\s+", header=None, names=COLUMNS)
    router = TacticalRULRouter()
    fleet = router.register_fleet_telemetry(df)
    decisions = router.solve_rul_vrptw(fleet)

    print("\n=== TAKTİK FİLO RUL ANALİZİ VE DİNAMİK ROTALAMA PLANI (RUL-VRPTW) ===")
    for d in decisions:
        print(
            f"{d['uav_id']} | RUL: {d['predicted_rul']:06.2f} | "
            f"Karar: {d['action']:<21} | Hedef Üs: {d['assigned_depot']:<14} | "
            f"Mesafe: {d['distance_km']:05.1f} km | Varış: {d['eta_min']:04.1f} dk"
        )

    os.makedirs(os.path.dirname(ROUTING_OUTPUT_PATH), exist_ok=True)
    with open(ROUTING_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(decisions, f, ensure_ascii=False, indent=2)
    print(f"\nRotalama planı kaydedildi -> {ROUTING_OUTPUT_PATH}")