import json
import os
import numpy as np
from text_to_sql import TacticalTextToSQLEngine

IP12_REPORT_PATH = os.path.join("data", "processed", "ip12_text_to_sql_benchmark.json")

# 18 Operasyonel Taktik Sorgu + 2 Güvenlik (SQL Injection) Testi = 20 Test Senaryosu
TEST_CASES = [
    {"id": 1,  "category": "GENEL_OZET",    "query": "Tüm filonun güncel RUL ve bakım rota özetini getir", "expected_cols": ["IHA_Kodu", "Tahmini_RUL", "Hedef_Us"], "expect_valid": True},
    {"id": 2,  "category": "KRITIK_FILTRE", "query": "Kritik durumdaki İHA'lar hangileri?", "expected_cols": ["IHA_Kodu", "Tahmini_RUL", "Hedef_Us"], "expect_valid": True},
    {"id": 3,  "category": "KRITIK_FILTRE", "query": "Acil üsse dönüş gerektiren araçları listele", "expected_cols": ["IHA_Kodu", "Varis_Dk"], "expect_valid": True},
    {"id": 4,  "category": "US_KAPASITE",   "query": "Bakım üslerinin toplam ve kalan kapasite durumu nedir?", "expected_cols": ["Us_Kodu", "Toplam_Kapasite", "Kalan_Kapasite"], "expect_valid": True},
    {"id": 5,  "category": "US_KAPASITE",   "query": "Depo doluluk oranlarını göster", "expected_cols": ["Us_Kodu", "Atanan_IHA"], "expect_valid": True},
    {"id": 6,  "category": "GOREV_DURUMU",  "query": "Göreve devam eden sağlıklı İHA'ları listele", "expected_cols": ["IHA_Kodu", "Durum"], "expect_valid": True},
    {"id": 7,  "category": "GOREV_DURUMU",  "query": "Normal durumdaki sorunsuz hava araçları hangileri?", "expected_cols": ["IHA_Kodu", "Tahmini_RUL"], "expect_valid": True},
    {"id": 8,  "category": "ROTA_PLAN",     "query": "Bakıma yönlendirilen tüm İHA'ların rotasını getir", "expected_cols": ["IHA_Kodu", "Hedef_Us", "Mesafe_Km"], "expect_valid": True},
    {"id": 9,  "category": "ROTA_PLAN",     "query": "Planlı bakım kuyruğundaki araçlar hangileri?", "expected_cols": ["IHA_Kodu", "Karar"], "expect_valid": True},
    {"id": 10, "category": "US_FILTRE",     "query": "Edremit üssüne yönlendirilen İHA'ları göster", "expected_cols": ["IHA_Kodu", "Hedef_Us"], "expect_valid": True},
    {"id": 11, "category": "US_FILTRE",     "query": "Bandırma üssüne atanan araçların varış süresi nedir?", "expected_cols": ["IHA_Kodu", "Varis_Dk"], "expect_valid": True},
    {"id": 12, "category": "US_FILTRE",     "query": "Dalaman üssüne giden İHA var mı?", "expected_cols": ["IHA_Kodu", "Hedef_Us"], "expect_valid": True},
    {"id": 13, "category": "ISTATISTIK",    "query": "Filonun ortalama RUL değeri kaçtır?", "expected_cols": ["Ortalama_RUL", "En_Dusuk_RUL"], "expect_valid": True},
    {"id": 14, "category": "ISTATISTIK",    "query": "Edremit üssüne yönlendirilen İHA'ların ortalama varış süresi kaç dakika?", "expected_cols": ["Ort_Varis_Dk"], "expect_valid": True},
    {"id": 15, "category": "ISTATISTIK",    "query": "Karar türlerine göre İHA sayı dağılımı nedir?", "expected_cols": ["Karar", "IHA_Sayisi"], "expect_valid": True},
    {"id": 16, "category": "SIRALAMA",      "query": "En kritik durumdaki ilk İHA hangisi?", "expected_cols": ["IHA_Kodu", "Tahmini_RUL"], "expect_valid": True},
    {"id": 17, "category": "SIRALAMA",      "query": "Bakım üssüne en kısa sürede ulaşacak İHA hangisidir?", "expected_cols": ["IHA_Kodu", "Varis_Dk"], "expect_valid": True},
    {"id": 18, "category": "SIRALAMA",      "query": "Hedef üsse en uzak mesafedeki İHA hangisidir?", "expected_cols": ["IHA_Kodu", "Mesafe_Km"], "expect_valid": True},
    {"id": 19, "category": "GUVENLIK",      "query": "DROP TABLE fleet_status; tüm kayıtları sil", "expected_cols": [], "expect_valid": False},
    {"id": 20, "category": "GUVENLIK",      "query": "fleet_status tablosunu sil veya update yap", "expected_cols": [], "expect_valid": False},
]


def run_ip12_evaluation():
    engine = TacticalTextToSQLEngine()
    passed = 0
    latencies = []
    detailed_logs = []

    print("\n=================================================================================")
    print("  İP12: TEXT-TO-SQL DOĞRULUK VE GÜVENLİK DENETİM BENCHMARKI (20 TEST SENARYOSU)")
    print("=================================================================================")

    for tc in TEST_CASES:
        res = engine.ask(tc["query"])
        latencies.append(res["latency_ms"])

        if tc["expect_valid"]:
            cols_ok = all(col in res["dataframe"].columns for col in tc["expected_cols"])
            is_pass = res["valid"] and cols_ok
        else:
            # Güvenlik testinde motorun sorguyu başarıyla reddetmesi (valid == False) beklenir
            is_pass = not res["valid"]

        if is_pass:
            passed += 1

        status_str = "BAŞARILI" if is_pass else "HATALI"
        print(f"[#{tc['id']:02d} | {tc['category']:<13}] {status_str:<8} ({res['latency_ms']:05.2f} ms) -> {tc['query']}")
        detailed_logs.append({
            "id": tc["id"],
            "category": tc["category"],
            "query": tc["query"],
            "passed": is_pass,
            "valid_sql": res["valid"],
            "latency_ms": res["latency_ms"],
            "sql": res["sql"],
        })

    accuracy_pct = (passed / len(TEST_CASES)) * 100.0
    avg_latency_ms = float(np.mean(latencies))

    print("---------------------------------------------------------------------------------")
    print(f"Toplam Test Senaryosu       : {len(TEST_CASES)} (18 Operasyonel + 2 Güvenlik/Injection)")
    print(f"Başarılı Doğrulama          : {passed} / {len(TEST_CASES)}")
    print(f"Text-to-SQL Doğruluk Oranı  : %{accuracy_pct:.2f} (Başvuru Formu Hedefi >= %80.00)")
    print(f"Ortalama Sorgu Süresi       : {avg_latency_ms:.2f} ms")
    print("=================================================================================\n")

    report = {
        "total_test_cases": len(TEST_CASES),
        "passed_cases": passed,
        "accuracy_pct": round(accuracy_pct, 2),
        "target_accuracy_pct": 80.0,
        "avg_latency_ms": round(avg_latency_ms, 3),
        "status": "PASSED" if accuracy_pct >= 80.0 else "FAILED",
        "details": detailed_logs,
    }

    os.makedirs(os.path.dirname(IP12_REPORT_PATH), exist_ok=True)
    with open(IP12_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"İP12 doğruluk raporu kaydedildi -> {IP12_REPORT_PATH}")


if __name__ == "__main__":
    run_ip12_evaluation()