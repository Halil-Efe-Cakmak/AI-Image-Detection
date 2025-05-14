import numpy as np
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.base import clone
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Load features and labels
X = np.load("features/X_features.npy")
y = np.load("features/y_labels.npy")

# Define models
models = {
    "SVM (RBF)": SVC(kernel='rbf', C=1, gamma='scale'),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, C=0.5),
    "KNN (k=11)": KNeighborsClassifier(n_neighbors=11),
    "MLP": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "LightGBM": LGBMClassifier(random_state=42),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42)
}

# Stratified K-Fold CV
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Sonuçları toplamak için
results = {}

# Loop through models
start_time = time.time()

for name, base_model in models.items():
    fold_accuracies = []
    all_y_true = []
    all_y_pred = []

    for train_index, test_index in kfold.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = clone(base_model)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        fold_accuracies.append(accuracy_score(y_test, y_pred))
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    results[name] = {
        "avg_accuracy": np.mean(fold_accuracies),
        "confusion_matrix": confusion_matrix(all_y_true, all_y_pred),
        "classification_report": classification_report(all_y_true, all_y_pred, output_dict=False)
    }

total_time = time.time() - start_time

# Tüm sonuçları yazdır
print("\n🔎 FINAL EVALUATION REPORT\n" + "=" * 35)

for name, metrics in results.items():
    print(f"\n📌 Model: {name}")
    print(f"✅ Average Accuracy: {metrics['avg_accuracy']:.4f}")
    print(f"📊 Confusion Matrix:\n{metrics['confusion_matrix']}")
    print(f"🧾 Classification Report:\n{metrics['classification_report']}")

print(f"\n⏲️ Total Evaluation Time: {total_time:.2f} seconds")
