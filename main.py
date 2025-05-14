import time
import os
import numpy as np
from feature_extraction_pipeline import process_dataset
from sklearn.preprocessing import StandardScaler

def main():
    start_time = time.time()
    ai_dir = os.path.join("data", "ai_generated")
    real_dir = os.path.join("data", "real_images")

    # Verileri çıkar
    X_ai, y_ai = process_dataset(ai_dir, label=1)
    X_real, y_real = process_dataset(real_dir, label=0)

    # X ve y'yi birleştir
    X = np.vstack((X_ai, X_real))
    y = np.concatenate((y_ai, y_real))

    # Normalizasyon
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Çıktı dosyalarını sil (varsa)
    if os.path.exists("features/X_features.npy"):
        os.remove("features/X_features.npy")
    if os.path.exists("features/y_labels.npy"):
        os.remove("features/y_labels.npy")

    # Yeni dosyaları kaydet
    np.save("features/X_features.npy", X_scaled)
    np.save("features/y_labels.npy", y)

    print("✅ Özellik çıkarımı ve dosyalar güncellendi.")
    end_time = time.time()
    duration = end_time - start_time
    print(f"⏱️ Toplam süre: {duration:.2f} saniye ({duration / 60:.2f} dakika)")

if __name__ == "__main__":
    main()
