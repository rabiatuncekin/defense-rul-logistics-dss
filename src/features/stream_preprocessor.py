import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType, MapType
)

# Kafka & Spark Yapılandırması
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC_NAME = "telemetry.tactical.raw"
SPARK_KAFKA_PACKAGE = "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1"

# JSON Telemetri Şeması
def get_telemetry_schema():
    return StructType([
        StructField("uav_id", StringType(), False),
        StructField("subsystem", StringType(), False),
        StructField("cycle", IntegerType(), False),
        StructField("timestamp", StringType(), False),
        StructField("operational_settings", StructType([
            StructField("setting_1", DoubleType(), True),
            StructField("setting_2", DoubleType(), True),
            StructField("setting_3", DoubleType(), True),
        ]), False),
        StructField("sensor_measurements", MapType(StringType(), DoubleType()), False)
    ])

def main():
    print("PySpark oturumu baslatiliyor...")
    
    # Java 17 / 21 / 25 sürümleri için modül erişim tanımları
    java_options = (
        "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
        "--add-opens=java.base/java.lang=ALL-UNNAMED "
        "--add-opens=java.base/java.util=ALL-UNNAMED"
    )

    spark = SparkSession.builder \
        .appName("Tactical-UAV-Feature-Pipeline") \
        .master("local[*]") \
        .config("spark.jars.packages", SPARK_KAFKA_PACKAGE) \
        .config("spark.driver.extraJavaOptions", java_options) \
        .config("spark.executor.extraJavaOptions", java_options) \
        .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", "true") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    print(f"Kafka'dan akis dinleniyor: {TOPIC_NAME} ({KAFKA_BOOTSTRAP_SERVERS})...")

    # 1. Kafka Stream Okuyucu (En baştan itibaren okur)
    raw_stream = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS) \
        .option("subscribe", TOPIC_NAME) \
        .option("startingOffsets", "earliest") \
        .load()

    schema = get_telemetry_schema()

    # 2. JSON Ayrıştırma ve Sensör Seçimi
    parsed_stream = raw_stream \
        .selectExpr("CAST(value AS STRING) as json_payload") \
        .select(from_json(col("json_payload"), schema).alias("data")) \
        .select("data.*")

    # Degradasyon analizi ve RUL tahmini için kritik 14 sensörün ayıklanması
    selected_features = parsed_stream.select(
        col("uav_id"),
        col("cycle"),
        col("operational_settings.setting_1").alias("setting_1"),
        col("operational_settings.setting_2").alias("setting_2"),
        col("sensor_measurements")["s_2"].alias("s_2_total_temp"),
        col("sensor_measurements")["s_3"].alias("s_3_total_temp_hpc"),
        col("sensor_measurements")["s_4"].alias("s_4_total_temp_lpt"),
        col("sensor_measurements")["s_7"].alias("s_7_hpc_pres"),
        col("sensor_measurements")["s_8"].alias("s_8_phys_core_speed"),
        col("sensor_measurements")["s_9"].alias("s_9_phys_fan_speed"),
        col("sensor_measurements")["s_11"].alias("s_11_static_hpc_pres"),
        col("sensor_measurements")["s_12"].alias("s_12_fuel_flow_ratio"),
        col("sensor_measurements")["s_13"].alias("s_13_corr_speed"),
        col("sensor_measurements")["s_14"].alias("s_14_corr_core_speed"),
        col("sensor_measurements")["s_15"].alias("s_15_bypass_ratio"),
        col("sensor_measurements")["s_17"].alias("s_17_bleed_enthalpy"),
        col("sensor_measurements")["s_20"].alias("s_20_hpt_cool_bleed"),
        col("sensor_measurements")["s_21"].alias("s_21_lpt_cool_bleed")
    )

    # 3. Konsola Canlı Akış Çıktısı
    query = selected_features.writeStream \
        .outputMode("append") \
        .format("console") \
        .option("truncate", "false") \
        .start()

    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        print("\nPySpark akisi sonlandirildi.")
        query.stop()
        spark.stop()

if __name__ == "__main__":
    main()