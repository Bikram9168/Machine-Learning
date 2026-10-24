import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier , plot_tree
from sklearn.metrics import accuracy_score

data = load_iris()
X=data.data
y=data.target

X_train , X_test , y_train , y_test = train_test_split(
    X , y , test_size=0.2 , random_state=42 , stratify=y
)

model = DecisionTreeClassifier( criterion="gini",max_depth=3,random_state=42)
model.fit(X_train,y_train)

y_pred=model.predict(X_test)
print(f"Accuracy Score")
