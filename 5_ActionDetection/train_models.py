import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
with open("data.pickle", "rb") as f:
    data_dict = pickle.load(f)

X_data = data_dict["data"]
y_data = data_dict["labels"]  # labels

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42, stratify=y_data)

# ML Models
models = {
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "SVM (RBF)": SVC(),
    "kNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression()
}

# Train and Evaluate each Model
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    results[name] = {"accuracy": acc, "f1_score": f1}

    print(f"--- {name} ---")
    print(f"Accuracy: {acc:.4f}, F1-score: {f1:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("\n")

    # --- Plot and Save Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{name.replace(' ', '_')}_confmatrix.png")
    plt.close()


# Compare results
print("=== Model Comparison ===")
for name, metrics in results.items():
    print(f"{name}: Accuracy={metrics['accuracy']:.4f}, F1-score={metrics['f1_score']:.4f}")