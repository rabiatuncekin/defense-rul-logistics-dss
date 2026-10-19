# Sistem Mimarisi ve Veri Akış Kurgusu (İP1)

## 1. Veri Akış Hattı
```text
[ C-MAPSS + Taktik Dinamikler ]
              │
              ▼
  [ Askeri Telemetri Simülatörü ]
              │ (JSON Paketleri / Hedef: >= 1000 msg/sec)
              ▼
    [ Apache Kafka Broker ]  ───> Topic: "telemetry.tactical.raw"
              │
              ▼
 [ PySpark Structured Streaming ]
   ├── Şema Doğrulama (Schema Enforcement)
   ├── Min-Max Normalizasyon & Sliding Window
   └── Mikro-Yığın Tamponlama
              │
              ▼
   [ LSTM Kestirimci Bakım Modeli ]
              │
              ├─── RUL > Eşik ──> [ TimescaleDB / PostgreSQL ]
              │                                        ▲
              └─── RUL <= Eşik (Kritik Alarm)          │
                         │                             │
                         ▼                             │
            [ Dinamik VRPTW Rota Motoru ]              │
            (NetworkX Çizge Optimizasyonu) ────────────┘
                         │
                         ▼
    [ Komuta Takip Paneli (Streamlit & Folium) ]
                         ▲
                         │
            [ Text-to-SQL / LLM Modülü ]
            (LangChain + Otomatik Kural Denetimi)