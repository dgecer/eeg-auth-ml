import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA


df = pd.read_csv("data/processed/features.csv")


print("\n=== LOADED FEATURES ===\n")

print(df.head())

print("\n=== SUBJECT DISTRIBUTION ===")

print(df["subject"].value_counts())


X = df.drop(columns=["subject"])

y = df["subject"]


scaler = StandardScaler()

X = scaler.fit_transform(X)


pca = PCA(n_components=20)

X = pca.fit_transform(X)


joblib.dump(scaler, "models/scaler.pkl")

joblib.dump(pca, "models/pca.pkl")


print("\n=== PCA APPLIED ===")

print(f"Reduced feature count: {X.shape[1]}")


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("\n=== TRAINING SVM MODEL ===\n")


model = SVC(
    kernel="rbf",
    probability=True
)


model.fit(X_train, y_train)


predictions = model.predict(X_test)


accuracy = accuracy_score(y_test, predictions)


print("\n=== MODEL ACCURACY ===\n")

print(f"Accuracy: {accuracy * 100:.2f}%")


cm = confusion_matrix(y_test, predictions)


print("\n=== CONFUSION MATRIX ===\n")

print(cm)


plt.figure(figsize=(8, 6))

plt.imshow(cm, cmap="RdPu")


plt.title("EEG Subject Classification Confusion Matrix")

plt.xlabel("Predicted Subject")

plt.ylabel("True Subject")


plt.colorbar()


classes = model.classes_


plt.xticks(range(len(classes)), classes)

plt.yticks(range(len(classes)), classes)


for i in range(len(classes)):

    for j in range(len(classes)):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            color="black"
        )


plt.tight_layout()


plt.savefig("results/confusion_matrix.png")


joblib.dump(model, "models/svm_model.pkl")


print("\nModel saved to models/svm_model.pkl")

print("\nScaler saved to models/scaler.pkl")

print("\nPCA saved to models/pca.pkl")

print("\nConfusion matrix saved to results/confusion_matrix.png")