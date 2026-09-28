import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

print("Student Dataset:")
print(df)

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
y_pred = model.predict(X_test_scaled)

print(f"Actual :{y_test.values}")
print(f"Prediction :{y_pred}")
print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_pred):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score  : {f1_score(y_test, y_pred):.4f}")

print("\nActual Values:")
print(y_test.values)

print("\nPredicted Values:")
print(y_pred)
