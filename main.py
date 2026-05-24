import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# ---------------- LOAD DATA ----------------

df = pd.read_csv("bank-full.csv", sep=';')

print("First 5 Rows:")
print(df.head())

# ---------------- CHECK DATA TYPES ----------------

print("\nData Types Before Encoding:")
print(df.dtypes)

# ---------------- ENCODE ALL TEXT COLUMNS ----------------

label_encoder = LabelEncoder()

for column in df.columns:
    if df[column].dtype == 'object' or df[column].dtype == 'str' or str(df[column].dtype) == 'string':
        df[column] = label_encoder.fit_transform(df[column])

# ---------------- VERIFY ENCODING ----------------

print("\nData Types After Encoding:")
print(df.dtypes)

# ---------------- FEATURES & TARGET ----------------

X = df.drop("y", axis=1)
y = df["y"]

# ---------------- TRAIN TEST SPLIT ----------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------- BUILD MODEL ----------------

model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

# ---------------- TRAIN MODEL ----------------

model.fit(X_train, y_train)

# ---------------- PREDICTIONS ----------------

y_pred = model.predict(X_test)

# ---------------- EVALUATION ----------------

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ---------------- CONFUSION MATRIX ----------------

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d'
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# ---------------- FEATURE IMPORTANCE ----------------

importance = model.feature_importances_

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

plt.figure(figsize=(10,6))

sns.barplot(
    data=feature_importance,
    x='Importance',
    y='Feature'
)

plt.title("Feature Importance")

plt.show()

# ---------------- DECISION TREE VISUALIZATION ----------------

plt.figure(figsize=(20,10))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["No", "Yes"],
    filled=True
)

plt.title("Decision Tree Classifier")

plt.show()