import os
import urllib.request
import pandas as pd

RAW_DATA_DIR = os.path.join("data", "raw")
TARGET_FILE = os.path.join(RAW_DATA_DIR, "train_FD001.txt")

# Çalışan alternatif C-MAPSS FD001 ayna adresleri
MIRROR_URLS = [
    "https://raw.githubusercontent.com/ashishpatel26/Turbofan-Engine-Degradation-Simulation-NASA-Data-Set/master/CMAPSSData/train_FD001.txt",
    "https://raw.githubusercontent.com/hankroark/Turbofan-Engine-Degradation/master/CMAPSSData/train_FD001.txt",
    "https://raw.githubusercontent.com/alewis5/NASA_Turbofan/master/train_FD001.txt"
]

COLUMNS = [
    "unit_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"
] + [f"sensor_{i}" for i in range(1, 22)]

def download_dataset():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    if not os.path.exists(TARGET_FILE):
        print("C-MAPSS FD001 veri seti indiriliyor...")
        downloaded = False
        
        for url in MIRROR_URLS:
            try:
                print(f"Deneniyor: {url}")
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req) as response, open(TARGET_FILE, "wb") as out_file:
                    out_file.write(response.read())
                print("İndirme başarıyla tamamlandı.")
                downloaded = True
                break
            except Exception as e:
                print(f"Bu kaynaktan indirilemedi: {e}. Alternatif deneniyor...")
        
        if not downloaded:
            raise RuntimeError("Hiçbir ayna sunucudan veri seti indirilemedi.")
    else:
        print(f"Veri seti zaten mevcut: {TARGET_FILE}")

    # İndirilen veriyi doğrulayalım
    df = pd.read_csv(TARGET_FILE, sep=r"\s+", header=None, names=COLUMNS)
    print(f"\nVeri Boyutu: {df.shape[0]} satır, {df.shape[1]} sütun.")
    print(f"Simüle Edilen Motor Sayısı: {df['unit_id'].nunique()}")

if __name__ == "__main__":
    download_dataset()