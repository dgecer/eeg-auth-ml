import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/processed/features.csv")
print("\n=== LOADED FEATURES ===\n")
print(df.head())

X = df[["delta", "theta", "alpha", "beta"]]
scaler = StandardScaler()
X = scaler.fit_transform(X)

y = df["subject"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\n=== TRAINING SVM MODEL ===\n")
model = SVC(kernel="linear")

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n=== SUBJECT DISTRIBUTION ===")
print(df["subject"].value_counts())

print("\n<=== MODEL ACCURACY ===>\n")
print(f"      Accuracy: {accuracy * 100:.2f}%")