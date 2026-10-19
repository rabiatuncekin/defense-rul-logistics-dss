# Sistem Analizi ve Gereksinim Belirtimi (İP1)

## 1. Fonksiyonel Gereksinimler
- FR-01: Simülatör, sentetik taktik verileri ve NASA C-MAPSS aşınma verilerini birleştirerek saniyede >= 1000 paket hızında Kafka'ya aktaracaktır.
- FR-02: PySpark Structured Streaming mikro-yığınlar halinde verileri okuyup şema denetimi ve normalizasyon yapacaktır.
- FR-03: PyTorch/TensorFlow LSTM modeli ile sensör serilerinden anlık RUL tahmini üretilecektir.
- FR-04: Kalan faydalı ömür kritik eşiğin altına indiğinde NetworkX tabanlı dinamik rota optimizasyonu (VRPTW) tetiklenecektir.
- FR-05: Streamlit arayüzünde canlı harita ve LLM destekli Text-to-SQL karar destek mekanizması sunulacaktır.

## 2. Başarı Kriterleri
- Veri akış gecikmesi: < 500 ms
- LSTM model başarımı: RMSE <= 15 çevrim, F1-Score >= %80
- Rota hesaplama süresi: < 2 saniye
- Text-to-SQL doğruluğu: >= %80