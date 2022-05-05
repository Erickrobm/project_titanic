import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from src.feature_engineering.transformers import transformers_categorical as tc
from src.feature_engineering.transformers import transformers_numerical as tn
from utils import config as cf

titanic_pipeline = Pipeline(
    [
        ("cabin_only_letter", tc.Cabin_Letter_Extractor()),
        ("drop_columns", tc.DropColumns()),
        ("missing_indicator", tn.Missing_Indicator(variables=cf.NUMERICAL_VARS)),
        (
            "categorical_imputer",
            tc.CategoricalImputerEncoder(variables=cf.CATEGORICAL_VARS),
        ),
        ("median_imputation", tn.MedianImputer(variables=cf.NUMERICAL_VARS)),
        ("rare_labels", tc.RareLabelEncoder(tol=0.05, variables=cf.CATEGORICAL_VARS)),
        ("dummy_vars", tc.OneHotEncoderImputer(variables=cf.CATEGORICAL_VARS)),
        ("scaling", MinMaxScaler()),
        (
            "log_reg",
            LogisticRegression(
                C=0.0005, class_weight="balanced", random_state=cf.SEED_MODEL
            ),
        ),
    ]
)
