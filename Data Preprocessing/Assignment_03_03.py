import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

np.random.seed(42)
data = {
    "Age": np.random.randint(22, 60, size=10).astype(float),
    "Salary": np.random.randint(40000, 120000, size=10).astype(float),
    "Department": np.random.choice(
        ["HR", "IT", "Marketing", "Sales"], size=10
    ),
    "Years_of_Experience": np.random.randint(1, 35, size=10).astype(
        float
    ),
}
df = pd.DataFrame(data)

df.loc[2, "Age"] = np.nan
df.loc[5, "Salary"] = np.nan
df.loc[4, "Department"] = np.nan
df.loc[7, "Years_of_Experience"] = np.nan

numeric_features = ["Age", "Salary", "Years_of_Experience"]
categorical_features = ["Department"]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)
transformed_data = preprocessor.fit_transform(df)
print("=== Transformed Array Output ===")
print(transformed_data.round(4))
