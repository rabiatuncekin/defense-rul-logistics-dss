import json
import math
import os
import sys
import time
import networkx as nx
import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.models.predict_rul import COLUMNS, RAW_DATA_PATH, RULPredictor

IP10_REPORT_PATH = os.path.join("data", "processed", "ip10_routing_benchmark.json")


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def build_base_depots() -> list[dict]:
    return [
        {
            "depot_id": "MOB-BANDIRMA",
            "name": "Bandırma Ana Harekat ve Ağır Bakım Üssü",
            "lat": 40.3178,
            "lon": 27.9775,
            "capacity": 3,
            "spare_modules": 4,
            "tw_start_min": 0,
            "tw_end_min": 180,
        },
        {
            "depot_id": "FMD-EDREMIT",
            "name": "Edremit İleri Bakım Noktası",
            "lat": 39.5546,
            "lon": 27.0138,
            "capacity": 2,
            "spare_modules": 2,
            "tw_start_min": 0,
            "tw_end_min": 120,
        },
        {
            "depot_id": "FMD-IZMIR",
            "name": "İzmir Taktik Destek Üssü",
            "lat": 38.2924,
            "lon": 27.1570,
            "capacity": 2,
            "spare_modules": 2,
            "tw_start_min": 0,
            "tw_end_min": 130,
        },
        {
            "depot_id": "FMD-DALAMAN",
            "name": "Dalaman İleri İkmal ve Bakım Üssü",
            "lat": 36.7131,
            "lon": 28.7925,
            "capacity": 2,
            "spare_modules": 2,
            "tw_start_min": 0,
            "tw_end_min": 150,
        },
    ]


def prepare_crisis_fleet(df: pd.DataFrame, predictor: RULPredictor) -> list[dict]:
    """NASA C-MAPSS train_FD001 motorlarından 12 İHA'lık çoklu kriz filosu oluşturur."""
    scenarios = [
        {"uav_id": "UAV-01", "unit_id": 1,  "cycle": 185, "lat": 39.65, "lon": 26.80, "cruise_kmh": 130},
        {"uav_id": "UAV-02", "unit_id": 2,  "cycle": 268, "lat": 39.40, "lon": 26.75, "cruise_kmh": 135},
        {"uav_id": "UAV-03", "unit_id": 3,  "cycle": 168, "lat": 39.20, "lon": 26.90, "cruise_kmh": 135},
        {"uav_id": "UAV-04", "unit_id": 4,  "cycle": 175, "lat": 38.60, "lon": 26.85, "cruise_kmh": 140},
        {"uav_id": "UAV-05", "unit_id": 5,  "cycle": 245, "lat": 38.45, "lon": 27.05, "cruise_kmh": 135},
        {"uav_id": "UAV-06", "unit_id": 6,  "cycle": 162, "lat": 39.75, "lon": 27.20, "cruise_kmh": 140},
        {"uav_id": "UAV-07", "unit_id": 7,  "cycle": 225, "lat": 37.90, "lon": 27.40, "cruise_kmh": 135},
        {"uav_id": "UAV-08", "unit_id": 8,  "cycle": 118, "lat": 37.30, "lon": 28.20, "cruise_kmh": 130},
        {"uav_id": "UAV-09", "unit_id": 9,  "cycle": 165, "lat": 37.05, "lon": 28.50, "cruise_kmh": 135},
        {"uav_id": "UAV-10", "unit_id": 10, "cycle": 60,  "lat": 40.10, "lon": 27.60, "cruise_kmh": 140},
        {"uav_id": "UAV-11", "unit_id": 11, "cycle": 70,  "lat": 38.90, "lon": 27.80, "cruise_kmh": 140},
        {"uav_id": "UAV-12", "unit_id": 12, "cycle": 50,  "lat": 37.60, "lon": 28.60, "cruise_kmh": 140},
    ]

    fleet = []
    for item in scenarios:
        unit_df = df[(df["unit_id"] == item["unit_id"]) & (df["cycle"] <= item["cycle"])].sort_values("cycle")
        pred = predictor.predict_from_window(unit_df)
        fleet.append({
            **item,
            "predicted_rul": pred["predicted_rul"],
            "status": pred["status"],
        })
    return fleet


def simulate_conventional_reactive(fleet: list[dict], depots: list[dict]) -> dict:
    """
    Yöntem 1 - Konvansiyonel Reaktif Planlama:
    Kestirimci bakım olmadan yalnızca RUL <= 12 (son dakika arıza uyarısı) olduğunda
    en yakın üsse yönlendirme yapılır; RUL 13..65 arası İHA'lar görevde bırakılır.
    """
    depot_loads = {d["depot_id"]: 0 for d in depots}
    depot_caps = {d["depot_id"]: d["capacity"] for d in depots}

    prevented_failures = 0
    in_flight_failures = 0
    total_turnaround_min = 0.0
    total_risk_exposure = 0.0

    for uav in fleet:
        rul = uav["predicted_rul"]
        if rul > 65:
            continue

        # Reaktif sistem RUL > 12 iken müdahale etmez; araç sahada kırıma/arızaya sürüklenir
        if rul > 12:
            in_flight_failures += 1
            total_turnaround_min += 240.0  # Sahadan kurtarma + ağır hasar cezası (dk)
            total_risk_exposure += (65.0 - rul) * 2.5
            continue

        # RUL <= 12: Son anda en yakın üsse yönlendir
        nearest = min(
            depots,
            key=lambda d: haversine_km(uav["lat"], uav["lon"], d["lat"], d["lon"])
        )
        dist_km = haversine_km(uav["lat"], uav["lon"], nearest["lat"], nearest["lon"])
        eta_min = (dist_km / uav["cruise_kmh"]) * 60.0
        queue_wait_min = max(0, depot_loads[nearest["depot_id"]] - depot_caps[nearest["depot_id"]] + 1) * 45.0
        depot_loads[nearest["depot_id"]] += 1

        required_cycles = (eta_min + queue_wait_min) / 15.0
        if required_cycles <= rul:
            prevented_failures += 1
            total_turnaround_min += eta_min + queue_wait_min
            total_risk_exposure += max(1.0, required_cycles / max(rul, 0.5)) * 20.0
        else:
            in_flight_failures += 1
            total_turnaround_min += 240.0
            total_risk_exposure += 120.0

    degraded_total = sum(1 for u in fleet if u["predicted_rul"] <= 65)
    return {
        "method": "Konvansiyonel Reaktif Planlama",
        "degraded_uavs": degraded_total,
        "prevented_failures": prevented_failures,
        "in_flight_failures": in_flight_failures,
        "safe_recovery_rate_pct": round((prevented_failures / degraded_total) * 100.0, 2),
        "avg_turnaround_min": round(total_turnaround_min / degraded_total, 2),
        "cumulative_risk_score": round(total_risk_exposure, 2),
    }


def simulate_classic_nearest_depot(fleet: list[dict], depots: list[dict]) -> dict:
    """
    Yöntem 2 - Klasik En Yakın Üs Rotalaması (Kapasite, Stok ve RUL Sağkalım Kısıtsız):
    Bakım ihtiyacı olan tüm İHA'ları en yakın üsse gönderir; yığılma nedeniyle
    bekleme süresi kalan ömrü aşarsa veya yedek parça biterse görev kaybı oluşur.
    """
    depot_loads = {d["depot_id"]: 0 for d in depots}
    depot_caps = {d["depot_id"]: d["capacity"] for d in depots}
    depot_spares = {d["depot_id"]: d["spare_modules"] for d in depots}

    prevented_failures = 0
    in_flight_failures = 0
    total_turnaround_min = 0.0
    total_risk_exposure = 0.0

    for uav in fleet:
        rul = uav["predicted_rul"]
        if rul > 65:
            continue

        nearest = min(
            depots,
            key=lambda d: haversine_km(uav["lat"], uav["lon"], d["lat"], d["lon"])
        )
        did = nearest["depot_id"]
        dist_km = haversine_km(uav["lat"], uav["lon"], nearest["lat"], nearest["lon"])
        eta_min = (dist_km / uav["cruise_kmh"]) * 60.0

        # Kapasite aşımında havada tur/kuyruk beklemesi + yedek parça yoksa aktarma gecikmesi
        overflow = max(0, depot_loads[did] - depot_caps[did] + 1)
        queue_wait_min = overflow * 50.0
        spare_penalty_min = 0.0 if depot_spares[did] > 0 else 75.0

        depot_loads[did] += 1
        if depot_spares[did] > 0:
            depot_spares[did] -= 1

        total_min = eta_min + queue_wait_min + spare_penalty_min
        required_cycles = total_min / 15.0

        if required_cycles <= rul and spare_penalty_min == 0.0:
            prevented_failures += 1
            total_turnaround_min += total_min
            total_risk_exposure += (required_cycles / max(rul, 1.0)) * 35.0
        else:
            in_flight_failures += 1
            total_turnaround_min += total_min + 90.0
            total_risk_exposure += 95.0

    degraded_total = sum(1 for u in fleet if u["predicted_rul"] <= 65)
    return {
        "method": "Klasik En Yakın Üs (Kısıtsız)",
        "degraded_uavs": degraded_total,
        "prevented_failures": prevented_failures,
        "in_flight_failures": in_flight_failures,
        "safe_recovery_rate_pct": round((prevented_failures / degraded_total) * 100.0, 2),
        "avg_turnaround_min": round(total_turnaround_min / degraded_total, 2),
        "cumulative_risk_score": round(total_risk_exposure, 2),
    }


def solve_proposed_rul_vrptw(fleet: list[dict], depots: list[dict]) -> tuple[dict, list[dict]]:
    """
    Yöntem 3 - Önerilen NetworkX RUL-VRPTW Optimizasyonu:
    En düşük RUL önceliği, RUL sağkalım marjı, zaman penceresi, bakım kapasitesi
    ve yedek modül stoğu kısıtları altında çizge tabanlı optimal atama yapar.
    """
    graph = nx.DiGraph()
    for d in depots:
        graph.add_node(d["depot_id"], node_type="DEPOT", **d)

    for uav in fleet:
        graph.add_node(uav["uav_id"], node_type="UAV", **uav)
        for d in depots:
            dist_km = haversine_km(uav["lat"], uav["lon"], d["lat"], d["lon"])
            eta_min = (dist_km / uav["cruise_kmh"]) * 60.0
            req_cycles = round(eta_min / 15.0, 2)
            graph.add_edge(
                uav["uav_id"],
                d["depot_id"],
                distance_km=round(dist_km, 2),
                eta_min=round(eta_min, 2),
                required_cycles=req_cycles,
            )

    rem_cap = {d["depot_id"]: d["capacity"] for d in depots}
    rem_spares = {d["depot_id"]: d["spare_modules"] for d in depots}
    safety_margin = 1.5

    sorted_uavs = sorted(fleet, key=lambda x: x["predicted_rul"])
    assignments = []
    prevented_failures = 0
    in_flight_failures = 0
    total_turnaround_min = 0.0
    total_risk_exposure = 0.0

    for uav in sorted_uavs:
        uid = uav["uav_id"]
        rul = uav["predicted_rul"]
        if rul > 65:
            assignments.append({
                "uav_id": uid,
                "predicted_rul": rul,
                "assigned_depot": "GÖREVDE_KAL",
                "eta_min": 0.0,
            })
            continue

        candidates = []
        for did in graph.successors(uid):
            d_node = graph.nodes[did]
            edge = graph.edges[uid, did]
            if rem_cap[did] <= 0 or rem_spares[did] <= 0:
                continue

            eta = edge["eta_min"]
            in_tw = d_node["tw_start_min"] <= eta <= d_node["tw_end_min"]
            survives = (edge["required_cycles"] + safety_margin) <= rul

            if in_tw and survives:
                # Çok amaçlı maliyet: intikal süresi + RUL risk oranı
                risk_ratio = edge["required_cycles"] / max(rul, 1.0)
                objective_cost = eta * (1.0 + risk_ratio)
                candidates.append((objective_cost, did, edge))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            _, best_did, best_edge = candidates[0]
            rem_cap[best_did] -= 1
            rem_spares[best_did] -= 1

            prevented_failures += 1
            total_turnaround_min += best_edge["eta_min"]
            total_risk_exposure += (best_edge["required_cycles"] / max(rul, 1.0)) * 20.0
            assignments.append({
                "uav_id": uid,
                "predicted_rul": rul,
                "assigned_depot": best_did,
                "eta_min": best_edge["eta_min"],
            })
        else:
            in_flight_failures += 1
            total_turnaround_min += 180.0
            total_risk_exposure += 90.0

    degraded_total = sum(1 for u in fleet if u["predicted_rul"] <= 65)
    summary = {
        "method": "Önerilen NetworkX RUL-VRPTW",
        "degraded_uavs": degraded_total,
        "prevented_failures": prevented_failures,
        "in_flight_failures": in_flight_failures,
        "safe_recovery_rate_pct": round((prevented_failures / degraded_total) * 100.0, 2),
        "avg_turnaround_min": round(total_turnaround_min / degraded_total, 2),
        "cumulative_risk_score": round(total_risk_exposure, 2),
    }
    return summary, assignments


def run_ip10_benchmark():
    df = pd.read_csv(RAW_DATA_PATH, sep=r"\s+", header=None, names=COLUMNS)
    predictor = RULPredictor()
    depots = build_base_depots()
    fleet = prepare_crisis_fleet(df, predictor)

    res_reactive = simulate_conventional_reactive(fleet, depots)
    res_nearest = simulate_classic_nearest_depot(fleet, depots)

    # Çözüm süresini 100 tekrar üzerinden ölç (Hedef < 2000 ms)
    solve_times_ms = []
    res_proposed = {}
    assignments = []
    for _ in range(100):
        t0 = time.perf_counter()
        res_proposed, assignments = solve_proposed_rul_vrptw(fleet, depots)
        solve_times_ms.append((time.perf_counter() - t0) * 1000.0)

    avg_solve_ms = float(np.mean(solve_times_ms))
    max_solve_ms = float(np.max(solve_times_ms))

    # Konvansiyonel reaktif planlamaya kıyasla operasyonel verimlilik ve risk azalımı
    risk_reduction_vs_reactive = (
        (res_reactive["cumulative_risk_score"] - res_proposed["cumulative_risk_score"])
        / res_reactive["cumulative_risk_score"]
    ) * 100.0
    turnaround_reduction_vs_reactive = (
        (res_reactive["avg_turnaround_min"] - res_proposed["avg_turnaround_min"])
        / res_reactive["avg_turnaround_min"]
    ) * 100.0
    risk_reduction_vs_nearest = (
        (res_nearest["cumulative_risk_score"] - res_proposed["cumulative_risk_score"])
        / res_nearest["cumulative_risk_score"]
    ) * 100.0

    print("\n=================================================================================")
    print("  İP10: RUL-VRPTW DİNAMİK ROTALAMA KARŞILAŞTIRMALI PERFORMANS ANALİZİ (12 İHA)")
    print("=================================================================================")
    for r in [res_reactive, res_nearest, res_proposed]:
        print(
            f"{r['method']:<32} | Kurtarılan: {r['prevented_failures']}/{r['degraded_uavs']} "
            f"(%{r['safe_recovery_rate_pct']:05.1f}) | Ort. İntikal: {r['avg_turnaround_min']:06.1f} dk | "
            f"Risk Skoru: {r['cumulative_risk_score']:06.1f}"
        )
    print("---------------------------------------------------------------------------------")
    print(f"RUL-VRPTW Ortalama Çözüm Süresi   : {avg_solve_ms:.3f} ms (Maks: {max_solve_ms:.3f} ms | Hedef < 2000 ms)")
    print(f"Konvansiyonele Göre Risk Azalımı  : %{risk_reduction_vs_reactive:.2f} (Hedef >= %80)")
    print(f"Konvansiyonele Göre Süre Kazancı  : %{turnaround_reduction_vs_reactive:.2f} (Hedef >= %80)")
    print(f"En Yakın Üs Yaklaşımına Göre Risk : %{risk_reduction_vs_nearest:.2f} daha düşük")
    print("=================================================================================\n")

    print("--- ÖNERİLEN RUL-VRPTW ÜS ATAMA DAĞILIMI ---")
    for a in assignments:
        print(f"  {a['uav_id']:<7} | Tahmini RUL: {a['predicted_rul']:06.2f} | Hedef: {a['assigned_depot']:<14} | Varış: {a['eta_min']:05.1f} dk")

    report = {
        "fleet_size": len(fleet),
        "degraded_uavs": res_proposed["degraded_uavs"],
        "solver_latency_avg_ms": round(avg_solve_ms, 4),
        "solver_latency_max_ms": round(max_solve_ms, 4),
        "target_latency_ms": 2000.0,
        "comparisons": {
            "conventional_reactive": res_reactive,
            "classic_nearest_depot": res_nearest,
            "proposed_rul_vrptw": res_proposed,
        },
        "efficiency_gains_pct": {
            "risk_reduction_vs_conventional": round(risk_reduction_vs_reactive, 2),
            "turnaround_reduction_vs_conventional": round(turnaround_reduction_vs_reactive, 2),
            "risk_reduction_vs_nearest_depot": round(risk_reduction_vs_nearest, 2),
        },
        "assignments": assignments,
    }

    os.makedirs(os.path.dirname(IP10_REPORT_PATH), exist_ok=True)
    with open(IP10_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nİP10 benchmark raporu kaydedildi -> {IP10_REPORT_PATH}")


if __name__ == "__main__":
    run_ip10_benchmark()