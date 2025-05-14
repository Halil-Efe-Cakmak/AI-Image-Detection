import numpy as np

X = np.load("features/X_features.npy")
y = np.load("features/y_labels.npy")

print("✅ X shape:", X.shape)
print("✅ y shape:", y.shape)

print("📌 All labels:", y)  # ← Tüm etiketleri yazdırır

# Örnek veri gösterimi (isteğe bağlı kalabilir)
print("🧪 First feature vector (first 10 values):", X[0][:10])

