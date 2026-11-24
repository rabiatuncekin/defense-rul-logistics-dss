import os
import sqlite3
import sys
import folium
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from streamlit_folium import st_folium

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.models.predict_rul import COLUMNS, RAW_DATA_PATH
from src.routing.rul_vrptw_router import TacticalRULRouter

DB_PATH = os.path.join("data", "processed", "tactical_dss.db")

st.set_page_config(
    page_title="C4ISR | İHA Kestirimci Bakım & Lojistik Karar Destek",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ÖZEL TAKTİK KOMUTA KONTROL (C4ISR) CSS TASARIMI ---
st.markdown("""
<style>
    /* Ana arka plan ve tipografi */
    .stApp {
        background-color: #090d16;
        color: #e2e8f0;
    }
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid rgba(56, 189, 248, 0.2);
    }
    /* Üst Komuta Başlık Banner'ı */
    .c4isr-header {
        background: linear-gradient(90deg, #0f172a 0%, #1e293b 60%, #0f172a 100%);
        border: 1px solid rgba(56, 189, 248, 0.35);
        border-left: 5px solid #38bdf8;
        border-radius: 10px;
        padding: 18px 24px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .c4isr-title {
        font-size: 1.55rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 0;
        letter-spacing: 0.4px;
    }
    .c4isr-subtitle {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 4px;
        font-family: 'Consolas', 'Courier New', monospace;
    }
    .live-badge {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        font-family: 'Consolas', monospace;
    }
    /* Özel KPI Kartları */
    .kpi-card {
        background: #111827;
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    .kpi-card.cyan { border-top: 3px solid #38bdf8; }
    .kpi-card.red { border-top: 3px solid #ef4444; }
    .kpi-card.amber { border-top: 3px solid #f59e0b; }
    .kpi-card.green { border-top: 3px solid #10b981; }
    .kpi-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #f8fafc;
        font-family: 'Consolas', 'Courier New', monospace;
    }
    .kpi-delta {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 4px;
    }
    /* Sekme (Tab) iyileştirmeleri */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0f172a;
        padding: 6px;
        border-radius: 8px;
        border: 1px solid rgba(148, 163, 184, 0.15);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 8px 18px;
        font-weight: 600;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e293b !important;
        color: #38bdf8 !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_raw_telemetry():
    return pd.read_csv(RAW_DATA_PATH, sep=r"\s+", header=None, names=COLUMNS)


def sync_tactical_database(fleet_df: pd.DataFrame, decisions_df: pd.DataFrame, depots_df: pd.DataFrame):
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    fleet_df.to_sql("fleet_status", conn, if_exists="replace", index=False)
    decisions_df.to_sql("routing_plan", conn, if_exists="replace", index=False)
    depots_df.to_sql("maintenance_depots", conn, if_exists="replace", index=False)
    conn.close()


def nl_to_sql_engine(question: str, crit_th: int) -> tuple[str, pd.DataFrame]:
    q = question.lower()

    if "kritik" in q or "acil" in q:
        sql = f"""
        SELECT f.uav_id AS IHA_Kodu, f.current_cycle AS Cevrim, f.predicted_rul AS Tahmini_RUL,
               r.action AS Karar, r.assigned_depot AS Hedef_Us, r.eta_min AS Varis_Dk
        FROM fleet_status f
        JOIN routing_plan r ON f.uav_id = r.uav_id
        WHERE f.predicted_rul <= {crit_th}
        ORDER BY f.predicted_rul ASC;
        """
    elif "edremit" in q or "bandırma" in q or "bandirma" in q or "dalaman" in q or "rota" in q:
        sql = """
        SELECT uav_id AS IHA_Kodu, predicted_rul AS Tahmini_RUL, action AS Karar,
               assigned_depot AS Hedef_Us, distance_km AS Mesafe_Km, eta_min AS Varis_Dk
        FROM routing_plan
        WHERE action = 'BAKIM ROTASI ATANDI'
        ORDER BY eta_min ASC;
        """
    elif "üs" in q or "kapasite" in q or "depo" in q:
        sql = """
        SELECT d.depot_id AS Us_Kodu, d.name AS Us_Adi, d.capacity AS Toplam_Kapasite,
               COUNT(r.uav_id) AS Atanan_IHA,
               (d.capacity - COUNT(r.uav_id)) AS Kalan_Kapasite
        FROM maintenance_depots d
        LEFT JOIN routing_plan r ON d.depot_id = r.assigned_depot
        GROUP BY d.depot_id, d.name, d.capacity;
        """
    elif "görev" in q or "normal" in q or "sağlıklı" in q:
        sql = """
        SELECT uav_id AS IHA_Kodu, current_cycle AS Cevrim, predicted_rul AS Tahmini_RUL, status AS Durum
        FROM fleet_status
        WHERE predicted_rul > 65;
        """
    else:
        sql = """
        SELECT f.uav_id AS IHA_Kodu, f.current_cycle AS Cevrim, f.predicted_rul AS Tahmini_RUL,
               r.action AS Karar, r.assigned_depot AS Hedef_Us, r.distance_km AS Mesafe_Km, r.eta_min AS Varis_Dk
        FROM fleet_status f
        JOIN routing_plan r ON f.uav_id = r.uav_id
        ORDER BY f.predicted_rul ASC;
        """

    conn = sqlite3.connect(DB_PATH)
    result_df = pd.read_sql_query(sql, conn)
    conn.close()
    return sql.strip(), result_df


def build_tactical_map(graph, decisions_df: pd.DataFrame, crit_th: int, warn_th: int) -> folium.Map:
    # API Key gerektirmeyen güvenilir OpenStreetMap + Esri Uydu katmanı
    m = folium.Map(location=[38.75, 27.65], zoom_start=7, tiles="OpenStreetMap", name="Taktik Harita")
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Uydu Görünümü",
        overlay=False,
        control=True
    ).add_to(m)

    # 1. Bakım Üsleri
    for node_id, data in graph.nodes(data=True):
        if data.get("type") == "DEPOT":
            folium.Marker(
                location=[data["lat"], data["lon"]],
                popup=folium.Popup(
                    f"<div style='font-family:sans-serif;min-width:180px'>"
                    f"<b>🔧 {data['name']}</b><br>"
                    f"Kapasite: <b>{data['capacity']} İHA</b><br>"
                    f"Hizmet: {data['service_level']}</div>",
                    max_width=250
                ),
                tooltip=f"ÜS: {node_id} (Kap: {data['capacity']})",
                icon=folium.Icon(color="cadetblue", icon="wrench", prefix="fa")
            ).add_to(m)

    # 2. İHA Konumları ve Bakım Rotaları
    decision_map = decisions_df.set_index("uav_id").to_dict(orient="index")
    for node_id, data in graph.nodes(data=True):
        if data.get("type") == "UAV":
            rul = data["predicted_rul"]
            dec = decision_map[node_id]

            if rul <= crit_th:
                color, line_color = "red", "#ef4444"
            elif rul <= warn_th:
                color, line_color = "orange", "#f59e0b"
            else:
                color, line_color = "green", "#10b981"

            folium.Marker(
                location=[data["lat"], data["lon"]],
                popup=folium.Popup(
                    f"<div style='font-family:sans-serif;min-width:190px'>"
                    f"<b>✈️ {node_id}</b><br>"
                    f"Aktif Çevrim: <b>{data['current_cycle']}</b><br>"
                    f"Tahmini RUL: <b>{rul} çevrim</b><br>"
                    f"Karar: <b>{dec['action']}</b><br>"
                    f"Hedef Üs: {dec['assigned_depot']} ({dec['eta_min']} dk)</div>",
                    max_width=260
                ),
                tooltip=f"{node_id} | RUL: {rul}",
                icon=folium.Icon(color=color, icon="plane", prefix="fa")
            ).add_to(m)

            target_depot = dec["assigned_depot"]
            if target_depot in graph.nodes:
                depot_data = graph.nodes[target_depot]
                folium.PolyLine(
                    locations=[[data["lat"], data["lon"]], [depot_data["lat"], depot_data["lon"]]],
                    color=line_color,
                    weight=3.5,
                    opacity=0.9,
                    dash_array="8, 6",
                    tooltip=f"{node_id} ➔ {target_depot} ({dec['distance_km']} km | {dec['eta_min']} dk)"
                ).add_to(m)

    folium.LayerControl(position="topright").add_to(m)
    return m


def main():
    # --- SOL KOMUTA KONTROL KENAR ÇUBUĞU (SIDEBAR) ---
    with st.sidebar:
        st.markdown("### ⚙️ Taktik Parametreler")
        st.caption("Kestirimci bakım eşiklerini ve sorti simülasyonunu anlık yönetin.")

        cycle_offset = st.slider(
            "⏱️ Sorti İlerletme (+Çevrim)",
            min_value=0,
            max_value=40,
            value=0,
            step=5,
            help="Filonun uçuş süresini ileri sararak LSTM modelinin yeni RUL tahminlerini ve rota değişimlerini gözlemleyin."
        )

        st.divider()
        crit_th = st.slider("🔴 Kritik RUL Eşiği (Acil Dönüş)", 15, 45, 30, step=5)
        warn_th = st.slider("🟠 Uyarı RUL Eşiği (Planlı Bakım)", 45, 90, 65, step=5)

        st.divider()
        st.markdown("**Mimari Katmanlar**")
        st.markdown(
            "- **Veri:** NASA C-MAPSS FD001\n"
            "- **Akış:** Kafka + PySpark\n"
            "- **Model:** 2-Katmanlı LSTM (`RMSE: 13.27`)\n"
            "- **Optimizasyon:** NetworkX RUL-VRPTW"
        )

    # --- ÜST KOMUTA BANNER ---
    st.markdown("""
    <div class="c4isr-header">
        <div>
            <p class="c4isr-title">🛡️ SAVUNMA LOJİSTİĞİ: KESTİRİMCİ BAKIM & DİNAMİK ROTALAMA KOKPİTİ</p>
            <div class="c4isr-subtitle">C-MAPSS TELEMETRİ • PYTORCH LSTM RUL • NETWORKX RUL-VRPTW • TEXT-TO-SQL</div>
        </div>
        <div class="live-badge">● SİSTEM AKTİF</div>
    </div>
    """, unsafe_allow_html=True)

    # --- VERİ VE MODEL ÇALIŞTIRMA ---
    df_raw = load_raw_telemetry()
    router = TacticalRULRouter()
    fleet_states = router.register_fleet_telemetry(df_raw)

    # Kenar çubuğundan sorti ilerletildiyse RUL ve eşikleri dinamik güncelle
    for uav in fleet_states:
        unit_all = df_raw[df_raw["unit_id"] == uav["unit_id"]].sort_values("cycle")
        max_avail_cycle = int(unit_all["cycle"].max())
        new_cycle = min(max_avail_cycle, uav["current_cycle"] + cycle_offset)

        if cycle_offset > 0:
            window_df = unit_all[unit_all["cycle"] <= new_cycle]
            pred = router.predictor.predict_from_window(window_df)
            uav["current_cycle"] = new_cycle
            uav["predicted_rul"] = pred["predicted_rul"]
            router.graph.nodes[uav["uav_id"]]["current_cycle"] = new_cycle
            router.graph.nodes[uav["uav_id"]]["predicted_rul"] = pred["predicted_rul"]

        rul = uav["predicted_rul"]
        if rul <= crit_th:
            uav["status"] = "KRİTİK"
        elif rul <= warn_th:
            uav["status"] = "UYARI"
        else:
            uav["status"] = "NORMAL"

    decisions = router.solve_rul_vrptw(fleet_states)
    fleet_df = pd.DataFrame(fleet_states)
    decisions_df = pd.DataFrame(decisions)
    depots_df = pd.DataFrame([
        {
            "depot_id": n,
            "name": d["name"],
            "capacity": d["capacity"],
            "service_level": d["service_level"],
            "tw_end_min": d["tw_end_min"],
            "lat": d["lat"],
            "lon": d["lon"]
        }
        for n, d in router.graph.nodes(data=True) if d.get("type") == "DEPOT"
    ])
    sync_tactical_database(fleet_df, decisions_df, depots_df)

    # --- ÖZEL KPI KARTLARI ---
    crit_count = int((fleet_df["predicted_rul"] <= crit_th).sum())
    routed_count = int((decisions_df["action"] == "BAKIM ROTASI ATANDI").sum())
    avg_rul = float(fleet_df["predicted_rul"].mean())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="kpi-card cyan">
            <div class="kpi-label">Aktif Taktik Filo</div>
            <div class="kpi-value">{len(fleet_df)} İHA</div>
            <div class="kpi-delta">Simülasyon Ofseti: +{cycle_offset} Çevrim</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card red">
            <div class="kpi-label">Kritik Eşik Altı (≤ {crit_th})</div>
            <div class="kpi-value">{crit_count} İHA</div>
            <div class="kpi-delta">Acil üsse dönüş öncelikli</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card amber">
            <div class="kpi-label">Bakım Rotası Atanan</div>
            <div class="kpi-value">{routed_count} / {len(fleet_df)}</div>
            <div class="kpi-delta">Kapasite & zaman penceresi uygun</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-card green">
            <div class="kpi-label">Ortalama Filo RUL</div>
            <div class="kpi-value">{avg_rul:.1f}</div>
            <div class="kpi-delta">Kalan faydalı ömür (çevrim)</div>
        </div>""", unsafe_allow_html=True)

    st.write("")

    # --- 3 SEKMELİ KOMUTA KONTROL DÜZENİ ---
    tab1, tab2, tab3 = st.tabs([
        "🗺️ Taktik Harita & Bakım Rotalama",
        "📈 Çoklu Sensör Degradasyon Analitiği",
        "💬 Komuta Kontrol Text-to-SQL"
    ])

    with tab1:
        map_col, table_col = st.columns([1.15, 1])

        with map_col:
            st.markdown("#### Harekat Alanı ve RUL-VRPTW Rotaları")
            tactical_map = build_tactical_map(router.graph, decisions_df, crit_th, warn_th)
            st_folium(tactical_map, width=None, height=470, returned_objects=[])

        with table_col:
            st.markdown("#### Filo Bakım Atama Matrisi")
            merged_df = fleet_df[["uav_id", "current_cycle", "predicted_rul", "status"]].merge(
                decisions_df[["uav_id", "action", "assigned_depot", "distance_km", "eta_min"]],
                on="uav_id"
            )
            st.dataframe(
                merged_df,
                use_container_width=True,
                hide_index=True,
                height=265,
                column_config={
                    "uav_id": st.column_config.TextColumn("İHA Kodu", width="medium"),
                    "current_cycle": st.column_config.NumberColumn("Çevrim", format="%d"),
                    "predicted_rul": st.column_config.ProgressColumn(
                        "Kalan Ömür (RUL)",
                        min_value=0,
                        max_value=125,
                        format="%.1f"
                    ),
                    "status": st.column_config.TextColumn("Durum"),
                    "action": st.column_config.TextColumn("Karar"),
                    "assigned_depot": st.column_config.TextColumn("Hedef Üs"),
                    "distance_km": st.column_config.NumberColumn("Mesafe", format="%.1f km"),
                    "eta_min": st.column_config.NumberColumn("Varış", format="%.1f dk"),
                }
            )

            st.markdown("#### Bakım Üssü Kapasite Durumu")
            depot_usage = decisions_df[decisions_df["action"] == "BAKIM ROTASI ATANDI"]["assigned_depot"].value_counts()
            depots_view = depots_df[["depot_id", "name", "capacity", "tw_end_min"]].copy()
            depots_view["atanan"] = depots_view["depot_id"].map(depot_usage).fillna(0).astype(int)
            depots_view["doluluk"] = (depots_view["atanan"] / depots_view["capacity"]) * 100
            st.dataframe(
                depots_view[["depot_id", "capacity", "atanan", "doluluk", "tw_end_min"]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "depot_id": "Üs Kodu",
                    "capacity": "Kapasite",
                    "atanan": "Atanan İHA",
                    "doluluk": st.column_config.ProgressColumn("Doluluk", min_value=0, max_value=100, format="%.0f%%"),
                    "tw_end_min": st.column_config.NumberColumn("Zaman Penceresi", format="0-%d dk")
                }
            )

    with tab2:
        sel_col, info_col = st.columns([1, 2])
        with sel_col:
            selected_uav = st.selectbox("Analiz Edilecek İHA:", fleet_df["uav_id"].tolist())
        uav_row = fleet_df[fleet_df["uav_id"] == selected_uav].iloc[0]
        with info_col:
            st.info(
                f"**{selected_uav}** | Mevcut Çevrim: **{uav_row['current_cycle']}** | "
                f"LSTM Tahmini RUL: **{uav_row['predicted_rul']} çevrim** | Durum: **{uav_row['status']}**"
            )

        uav_hist = df_raw[
            (df_raw["unit_id"] == int(uav_row["unit_id"])) &
            (df_raw["cycle"] <= int(uav_row["current_cycle"]))
        ]

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "LPT Çıkış Sıcaklığı (sensor_4 - °R)",
                "HPC Statik Basıncı (sensor_11 - psia)",
                "Yakıt Akış Oranı (sensor_12 - pps/psi)",
                "HPC Toplam Basıncı (sensor_7 - psia)"
            )
        )
        fig.add_trace(go.Scatter(x=uav_hist["cycle"], y=uav_hist["sensor_4"], line=dict(color="#ef4444", width=2.5), name="s4"), row=1, col=1)
        fig.add_trace(go.Scatter(x=uav_hist["cycle"], y=uav_hist["sensor_11"], line=dict(color="#f59e0b", width=2.5), name="s11"), row=1, col=2)
        fig.add_trace(go.Scatter(x=uav_hist["cycle"], y=uav_hist["sensor_12"], line=dict(color="#38bdf8", width=2.5), name="s12"), row=2, col=1)
        fig.add_trace(go.Scatter(x=uav_hist["cycle"], y=uav_hist["sensor_7"], line=dict(color="#10b981", width=2.5), name="s7"), row=2, col=2)

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#111827",
            plot_bgcolor="#0f172a",
            height=460,
            showlegend=False,
            margin=dict(l=20, r=20, t=45, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("#### Doğal Dilden SQL Üreten Taktik Sorgu Motoru")
        q_col1, q_col2 = st.columns([1.2, 1])
        with q_col1:
            sample_questions = [
                "Tüm filonun güncel RUL ve bakım rota özetini getir",
                "Kritik durumdaki İHA'lar hangileri?",
                "Bakım üslerinin toplam ve kalan kapasite durumu nedir?",
                "Göreve devam eden sağlıklı İHA'ları listele"
            ]
            selected_prompt = st.selectbox("Hazır Komuta Sorguları:", sample_questions)
        with q_col2:
            user_query = st.text_input("Veya Özel Sorgu Yazın:", value=selected_prompt)

        if st.button("⚡ Sorguyu Yürüt", type="primary"):
            generated_sql, sql_result = nl_to_sql_engine(user_query, crit_th)
            with st.expander("Üretilen SQL Sorgusu", expanded=True):
                st.code(generated_sql, language="sql")
            st.dataframe(sql_result, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()