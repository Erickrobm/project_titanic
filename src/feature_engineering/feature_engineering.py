import pandas as pd
import numpy as np
from tabnanny import verbose
from transformers import transformers_categorical as tc
from transformers import transformers_numerical as tn
from data.cleaned.utils import config
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

train_df = pd.read_csv("data/cleaned/train.csv")
test_df = pd.read_csv("data/cleaned/test.csv")

# Categorical variables pipeline
categorical_imputer_encoder = tc.CategoricalImputerEncoder(config.CATEGORICAL_VARS)
rare_label_encoder = tc.RareLabelEncoder(variables=["cabin"])
one_hot_encoder = ColumnTransformer(
    ["ohe", OneHotEncoder(drop="first"), config.CATEGORICAL_VARS],
    remainder="passthrough",
)

categorical_pipe = Pipeline(
    [
        ("cie", categorical_imputer_encoder),
        ("rle", rare_label_encoder),
        ("ohe", one_hot_encoder),
    ]
)

categorical_pipe.fit(train_df)

processed_train_df = categorical_pipe.transform(train_df)
processed_test_df = categorical_pipe.transform(test_df)

# Numerical variables pipeline
missing_indicator = tn.Missing_Indicator(config.NUMERICAL_VARS)
numerical_median_imputer = SimpleImputer(strategy="median")
min_max_scaler = MinMaxScaler()

numerical_pipe = Pipeline(
    [
        ("mi", missing_indicator),
        ("nmi", numerical_median_imputer),
        ("mms", min_max_scaler),
    ]
)

numeric_col_transformer = ColumnTransformer(
    [("numeric_transforms", numerical_pipe, config.NUMERICAL_VARS)]
)

processed_train_df[config.NUMERICAL_VARS] = numeric_col_transformer.fit_transform(
    processed_train_df
)

processed_test_df[config.NUMERICAL_VARS] = numeric_col_transformer.fit_transform(
    processed_test_df
)

processed_test_df = processed_test_df[processed_train_df.columns]

# Save processed DFs
processed_train_df.to_csv(config.PROCESSED_TRAIN_DATA_FILE, index=False)
processed_test_df.to_csv(config.PROCESSED_TEST_DATA_FILE, index=False)
