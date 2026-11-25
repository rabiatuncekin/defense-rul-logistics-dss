import os
import re
import sqlite3
import sys
import time
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.models.predict_rul import COLUMNS, RAW_DATA_PATH
from src.routing.rul_vrptw_router import TacticalRULRouter

DB_PATH = os.path.join("data", "processed", "tactical_dss.db")

ALLOWED_TABLES = {
    "fleet_status",
    "routing_plan",
    "maintenance_depots",
    "live_stream_predictions",
}

FORBIDDEN_KEYWORDS = {
    "DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE",
    "REPLACE", "TRUNCATE", "ATTACH", "DETACH", "PRAGMA", "VACUUM", "EXEC"
}


class SQLSafetyValidator:
    """Üretilen SQL sorgularını güvenlik, yetkili tablo ve sözdizimi kurallarına göre denetler."""

    @staticmethod
    def validate(sql: str, db_path: str = DB_PATH) -> tuple[bool, str]:
        if not sql or not isinstance(sql, str):
            return False, "Boş SQL sorgusu üretildi."

        cleaned = sql.strip().rstrip(";").strip()
        upper_sql = cleaned.upper()

        if not (upper_sql.startswith("SELECT") or upper_sql.startswith("WITH")):
            return False, "Güvenlik İhlali: Yalnızca salt-okunur SELECT sorgularına izin verilir."

        if ";" in cleaned:
            return False, "Güvenlik İhlali: Çoklu SQL ifadesi tespit edildi."

        tokens = set(re.findall(r"\b[A-Z_]+\b", upper_sql))
        blocked = tokens.intersection(FORBIDDEN_KEYWORDS)
        if blocked:
            return False, f"Güvenlik İhlali: Yasaklı SQL komutları tespit edildi -> {sorted(blocked)}"

        referenced_tables = re.findall(r"\b(?:FROM|JOIN)\s+([a-zA-Z0-9_]+)", cleaned, flags=re.IGNORECASE)
        if not referenced_tables:
            return False, "Şema Hatası: Sorguda geçerli bir tablo referansı bulunamadı."

        for tbl in referenced_tables:
            if tbl.lower() not in ALLOWED_TABLES:
                return False, f"Yetkisiz Tablo Erişimi: '{tbl}' izin verilen şemada yer almıyor."

        try:
            conn = sqlite3.connect(db_path)
            conn.execute(f"EXPLAIN QUERY PLAN {cleaned}")
            conn.close()
        except sqlite3.Error as exc:
            return False, f"SQL Sözdizimi/Şema Hatası: {exc}"

        return True, "VALID"


class TacticalTextToSQLEngine:
    """Komuta kontrol doğal dil sorularını güvenli SQL sorgularına dönüştüren hibrit motor."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._ensure_database_ready()
        self.llm_chain = self._init_langchain_if_available()

    def _ensure_database_ready(self):
        if os.path.exists(self.db_path):
            conn = sqlite3.connect(self.db_path)
            cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='fleet_status';")
            exists = cur.fetchone() is not None
            conn.close()
            if exists:
                return

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        df_raw = pd.read_csv(RAW_DATA_PATH, sep=r"\s+", header=None, names=COLUMNS)
        router = TacticalRULRouter()
        fleet = router.register_fleet_telemetry(df_raw)
        decisions = router.solve_rul_vrptw(fleet)

        fleet_df = pd.DataFrame(fleet)
        decisions_df = pd.DataFrame(decisions)
        depots_df = pd.DataFrame([
            {
                "depot_id": n,
                "name": d["name"],
                "capacity": d["capacity"],
                "service_level": d["service_level"],
                "tw_end_min": d["tw_end_min"],
                "lat": d["lat"],
                "lon": d["lon"],
            }
            for n, d in router.graph.nodes(data=True)
            if d.get("type") == "DEPOT"
        ])

        conn = sqlite3.connect(self.db_path)
        fleet_df.to_sql("fleet_status", conn, if_exists="replace", index=False)
        decisions_df.to_sql("routing_plan", conn, if_exists="replace", index=False)
        depots_df.to_sql("maintenance_depots", conn, if_exists="replace", index=False)
        conn.close()

    def _init_langchain_if_available(self):
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            return None
        try:
            from langchain_community.utilities import SQLDatabase
            from langchain_openai import ChatOpenAI

            db = SQLDatabase.from_uri(f"sqlite:///{self.db_path}")
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            return (llm, db)
        except Exception:
            return None

    def _semantic_nl_to_sql(self, question: str) -> str:
        """Şema farkındalıklı anlamsal kural + varlık çıkarımı ile deterministik SQL üretir."""
        q = question.lower().strip()

        # Güvenlik testi için zararlı niyet varsa ham komutu döndür (Validator yakalasın)
        if any(bad in q for bad in ["drop table", "delete from", "sil ", "tabloyu sil", "update "]):
            return "DROP TABLE fleet_status;"

        # 1. Üs bazlı spesifik filtrelemeler (Edremit / Bandırma / Dalaman)
        for depot_kw, depot_code in [
            ("edremit", "FMD-EDREMIT"),
            ("bandırma", "MOB-BANDIRMA"),
            ("bandirma", "MOB-BANDIRMA"),
            ("dalaman", "FMD-DALAMAN"),
        ]:
            if depot_kw in q:
                if "ortalama" in q:
                    return (
                        f"SELECT assigned_depot AS Hedef_Us, COUNT(uav_id) AS IHA_Sayisi, "
                        f"ROUND(AVG(distance_km), 2) AS Ort_Mesafe_Km, ROUND(AVG(eta_min), 2) AS Ort_Varis_Dk "
                        f"FROM routing_plan WHERE assigned_depot = '{depot_code}' GROUP BY assigned_depot;"
                    )
                return (
                    f"SELECT uav_id AS IHA_Kodu, predicted_rul AS Tahmini_RUL, action AS Karar, "
                    f"assigned_depot AS Hedef_Us, distance_km AS Mesafe_Km, eta_min AS Varis_Dk "
                    f"FROM routing_plan WHERE assigned_depot = '{depot_code}' ORDER BY eta_min ASC;"
                )

        # 2. Toplulaştırma / İstatistik / Ortalama soruları
        if "ortalama" in q and ("rul" in q or "ömür" in q):
            return (
                "SELECT COUNT(uav_id) AS Toplam_IHA, ROUND(AVG(predicted_rul), 2) AS Ortalama_RUL, "
                "MIN(predicted_rul) AS En_Dusuk_RUL, MAX(predicted_rul) AS En_Yuksek_RUL "
                "FROM fleet_status;"
            )

        if ("kaç" in q or "sayı" in q or "dağılım" in q) and ("karar" in q or "durum" in q or "bakım" in q):
            return (
                "SELECT action AS Karar, COUNT(uav_id) AS IHA_Sayisi, ROUND(AVG(predicted_rul), 2) AS Ortalama_RUL "
                "FROM routing_plan GROUP BY action ORDER BY IHA_Sayisi DESC;"
            )

        # 3. En yakın / En uzak / En hızlı / En kritik sıralamalı sorgular
        if "en kritik" in q or "en düşük rul" in q or "en az ömrü" in q or "ilk önce" in q:
            return (
                "SELECT f.uav_id AS IHA_Kodu, f.current_cycle AS Cevrim, f.predicted_rul AS Tahmini_RUL, "
                "r.assigned_depot AS Hedef_Us, r.eta_min AS Varis_Dk "
                "FROM fleet_status f JOIN routing_plan r ON f.uav_id = r.uav_id "
                "ORDER BY f.predicted_rul ASC LIMIT 1;"
            )

        if "en uzak" in q or "en uzun" in q:
            return (
                "SELECT uav_id AS IHA_Kodu, assigned_depot AS Hedef_Us, distance_km AS Mesafe_Km, eta_min AS Varis_Dk "
                "FROM routing_plan WHERE action = 'BAKIM ROTASI ATANDI' "
                "ORDER BY distance_km DESC LIMIT 1;"
            )

        if "en yakın" in q or "en hızlı" in q or "en kısa" in q:
            return (
                "SELECT uav_id AS IHA_Kodu, assigned_depot AS Hedef_Us, distance_km AS Mesafe_Km, eta_min AS Varis_Dk "
                "FROM routing_plan WHERE action = 'BAKIM ROTASI ATANDI' "
                "ORDER BY eta_min ASC LIMIT 1;"
            )

        # 4. Kritik / Acil bakım eşiği sorguları (Genel üs kapasite sorgularından önce kontrol edilir)
        if "kritik" in q or "acil" in q or "risk" in q:
            return (
                "SELECT f.uav_id AS IHA_Kodu, f.current_cycle AS Cevrim, f.predicted_rul AS Tahmini_RUL, "
                "r.action AS Karar, r.assigned_depot AS Hedef_Us, r.eta_min AS Varis_Dk "
                "FROM fleet_status f JOIN routing_plan r ON f.uav_id = r.uav_id "
                "WHERE f.predicted_rul <= 30 ORDER BY f.predicted_rul ASC;"
            )

        # 5. Kapasite ve bakım üssü sorguları
        if "üs" in q or "kapasite" in q or "depo" in q or "doluluk" in q:
            return (
                "SELECT d.depot_id AS Us_Kodu, d.name AS Us_Adi, d.capacity AS Toplam_Kapasite, "
                "COUNT(r.uav_id) AS Atanan_IHA, (d.capacity - COUNT(r.uav_id)) AS Kalan_Kapasite "
                "FROM maintenance_depots d LEFT JOIN routing_plan r ON d.depot_id = r.assigned_depot "
                "GROUP BY d.depot_id, d.name, d.capacity ORDER BY Kalan_Kapasite ASC;"
            )

        # 6. Uyarı / Planlı bakım kuyruğu sorguları
        if "uyarı" in q or "planlı" in q or "bakıma yönlendirilen" in q or "rota" in q:
            return (
                "SELECT uav_id AS IHA_Kodu, predicted_rul AS Tahmini_RUL, action AS Karar, "
                "assigned_depot AS Hedef_Us, distance_km AS Mesafe_Km, eta_min AS Varis_Dk "
                "FROM routing_plan WHERE action = 'BAKIM ROTASI ATANDI' ORDER BY predicted_rul ASC;"
            )

        # 7. Göreve devam eden sağlıklı İHA'lar
        if "görev" in q or "normal" in q or "sağlıklı" in q or "sorunsuz" in q:
            return (
                "SELECT uav_id AS IHA_Kodu, current_cycle AS Cevrim, predicted_rul AS Tahmini_RUL, status AS Durum "
                "FROM fleet_status WHERE predicted_rul > 65 ORDER BY predicted_rul DESC;"
            )

        # 8. Canlı PySpark akış tablosu sorguları
        if "canlı" in q or "akış" in q or "spark" in q or "çıkarım" in q:
            return (
                "SELECT uav_id AS IHA_Kodu, cycle AS Cevrim, predicted_rul AS Canli_RUL, "
                "status AS Durum, inference_ms AS Cikarim_Ms "
                "FROM live_stream_predictions ORDER BY predicted_rul ASC;"
            )

        # 9. Varsayılan genel filo ve rota özeti
        return (
            "SELECT f.uav_id AS IHA_Kodu, f.current_cycle AS Cevrim, f.predicted_rul AS Tahmini_RUL, "
            "r.action AS Karar, r.assigned_depot AS Hedef_Us, r.distance_km AS Mesafe_Km, r.eta_min AS Varis_Dk "
            "FROM fleet_status f JOIN routing_plan r ON f.uav_id = r.uav_id "
            "ORDER BY f.predicted_rul ASC;"
        )

    def ask(self, question: str) -> dict:
        t0 = time.perf_counter()
        engine_mode = "LangChain-LLM" if self.llm_chain else "Schema-Semantic-Engine"

        if self.llm_chain:
            llm, db = self.llm_chain
            prompt = (
                f"Sen bir askeri lojistik SQL uzmanısın. Sadece geçerli bir SQLite SELECT sorgusu döndür, "
                f"açıklama yazma.\nŞema:\n{db.get_table_info()}\nSoru: {question}"
            )
            raw_resp = llm.invoke(prompt).content
            generated_sql = raw_resp.replace("```sql", "").replace("```", "").strip()
        else:
            generated_sql = self._semantic_nl_to_sql(question)

        is_valid, validation_msg = SQLSafetyValidator.validate(generated_sql, self.db_path)
        if not is_valid:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "question": question,
                "engine": engine_mode,
                "sql": generated_sql,
                "valid": False,
                "validation_message": validation_msg,
                "latency_ms": round(elapsed_ms, 3),
                "dataframe": pd.DataFrame(),
            }

        conn = sqlite3.connect(self.db_path)
        df_res = pd.read_sql_query(generated_sql, conn)
        conn.close()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "question": question,
            "engine": engine_mode,
            "sql": generated_sql,
            "valid": True,
            "validation_message": "VALID",
            "latency_ms": round(elapsed_ms, 3),
            "dataframe": df_res,
        }


if __name__ == "__main__":
    engine = TacticalTextToSQLEngine()
    test_q = "Kritik durumdaki İHA'lar hangileri ve hangi üsse yönlendirildi?"
    out = engine.ask(test_q)
    print(f"Soru   : {out['question']}")
    print(f"Motor  : {out['engine']} | Durum: {out['validation_message']} | Süre: {out['latency_ms']} ms")
    print(f"SQL    : {out['sql']}")
    print(out["dataframe"].to_string(index=False))