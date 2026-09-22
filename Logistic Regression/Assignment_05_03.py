import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

data = {
    "Study_Hours": [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        2, 3, 4, 5, 6, 7, 8, 9, 10, 11
    ],

    "Attendance": [
        50, 55, 60, 65, 70, 72, 75, 80, 85, 90,
        52, 58, 63, 68, 73, 78, 82, 87, 92, 95
    ],

    "Previous_Exam_Score": [
        35, 40, 42, 48, 55, 60, 65, 70, 78, 85,
        38, 44, 50, 58, 63, 68, 75, 80, 88, 92
    ],

    "Assignments_Completed": [
        3, 4, 5, 6, 7, 8, 8, 9, 10, 10,
        4, 5, 6, 7, 8, 9, 9, 10, 10, 10
    ],

    "Sleep_Hours": [
        5, 5.5, 6, 6, 7, 7, 7.5, 8, 8, 8,
        5, 6, 6.5, 7, 7, 7.5, 8, 8, 8.5, 9
    ],

    "Participation": [
        20, 25, 30, 35, 45, 50, 55, 65, 75, 85,
        22, 30, 40, 50, 55, 60, 70, 78, 88, 92
    ],

    "Pass": [
        0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 1, 1, 1, 1, 1, 1, 1
    ]
}

df = pd.DataFrame(data)

X = df[
    [
        "Study_Hours",
        "Attendance",
        "Previous_Exam_Score",
        "Assignments_Completed",
        "Sleep_Hours",
        "Participation"
    ]
]

y = df["Pass"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)

model.fit(X_train_scaled, y_train)

y_probability = model.predict_proba(X_test_scaled)[:, 1]

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

auc_score = roc_auc_score(
    y_test,
    y_probability
)

print(f"AUC Score: {auc_score:.4f}")

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"Logistic Regression (AUC = {auc_score:.4f})"
)
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve - Student Pass/Fail Prediction")

plt.legend()

plt.grid()

plt.show()