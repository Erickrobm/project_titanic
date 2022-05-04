import pandas as pd
from tabnanny import verbose
from transformers import transformers_categorical as tc
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# Find numeric variables and categorical variables in data
numeric_vars = [
    col
    for col in train_df.columns
    if train_df[col].dtype != object and col != "survived"
]
categorical_vars = [
    col
    for col in train_df.columns
    if train_df[col].dtype == object and col != "survived"
]

# Categorical variables pipeline
missing_imputer = tc.MissingImputerEncoder(variables=categorical_vars)
rare_label_encoder = tc.RareLabelEncoder(variables=["cabin"])

categorical_pipe = Pipeline([("mi", missing_imputer), ("rle", rare_label_encoder)])
categorical_pipe.fit(train_df)

processed_train_df = categorical_pipe.transform(train_df)
processed_test_df = categorical_pipe.transform(test_df)

# Numerical variables pipeline
numerical_median_imputer = SimpleImputer(strategy="median")
min_max_scaler = MinMaxScaler()

ohe = ColumnTransformer(
    ["ohe", OneHotEncoder(drop="first"), categorical_vars], remainder="passthrough"
)
nct = ColumnTransformer(
    [
        ("nmi", numerical_median_imputer, ["age", "fare"]),
        ("mms", min_max_scaler, ["age", "fare"]),
    ],
    remainder="passthrough",
)

numerical_pipe = Pipeline([("numeric_transform", nct), ("onehot", ohe)])
numerical_pipe.fit(processed_train_df)
xxx = numerical_pipe.transform(processed_test_df)

print(xxx[0:3])
