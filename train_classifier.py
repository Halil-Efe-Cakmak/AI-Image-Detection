import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Boosting modelleri (önce pip ile kur: xgboost, lightgbm, catboost)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# --- AŞAMA 1: VERİYİ YÜKLE ---

train_X = np.load("train_X.npy")
train_y = np.load("train_y.npy")
val_X = np.load("val_X.npy")
val_y = np.load("val_y.npy")
test_X = np.load("test_X.npy")
test_y = np.load("test_y.npy")

print(f"Train: {train_X.shape}, Validation: {val_X.shape}, Test: {test_X.shape}")

# --- AŞAMA 2: MODEL LİSTESİ ---

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Gradient Boosting": GradientBoostingClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "LightGBM": LGBMClassifier(),
    "CatBoost": CatBoostClassifier(verbose=0),
    "MLP (Neural Net)": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
}

# --- AŞAMA 3: MODEL EĞİTİMİ VE DEĞERLENDİRME ---

for name, model in models.items():
    print(f"\n🔍 {name} eğitiliyor...")
    model.fit(train_X, train_y)

    val_acc = model.score(val_X, val_y)
    print(f"✅ Validation Accuracy: {val_acc:.4f}")

    # Test performansı
    test_preds = model.predict(test_X)
    test_acc = accuracy_score(test_y, test_preds)

    print(f"✅ Test Accuracy: {test_acc:.4f}")
    print("📊 Classification Report (Test Set):")
    print(classification_report(test_y, test_preds, target_names=["fake", "real"]))
    print("📉 Confusion Matrix (Test Set):")
    print(confusion_matrix(test_y, test_preds))

    # Modeli kaydet
    filename = name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".pkl"
    joblib.dump(model, filename)
    print(f"💾 Model kaydedildi: {filename}")
