import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



data = {
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                    2, 3, 4, 5, 6, 7, 8, 9, 10, 11],

    "Attendance": [50, 55, 60, 65, 70, 72, 75, 80, 85, 90,
                52, 58, 63, 68, 73, 78, 82, 87, 92, 95],

    "Pass": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
                0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)
print(df)
X= df[["Study_Hours","Attendance"]]
y = df["Pass"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
random_state=42, stratify=y)


model  = LogisticRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print(f"Accuracy:{accuracy_score(y_test,y_pred):.4f}")
print(f"Precision:{precision_score(y_test,y_pred):.4f}")
print(f"Recall:{recall_score(y_test,y_pred):.4f}")
print(f"F1 Score : {f1_score(y_test,y_pred):.4f}")
print(f"Actual :{y_test.values}")
print(f"Prediction :{y_pred}")