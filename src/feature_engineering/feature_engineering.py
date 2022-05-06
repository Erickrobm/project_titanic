import pandas as pd
from transformers import transformers_categorical as tc
from transformers import transformers_numerical as tn
from utils import config
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

train_df = pd.read_csv(config.TRAIN_DATA_FILE)
test_df = pd.read_csv(config.TEST_DATA_FILE)

# Numerical and Categorical variables without persisting information - PIPELINE
vars_without_persisting_info = Pipeline(
    [
        ("missing_indicator", tn.Missing_Indicator(config.NUMERICAL_VARS)),
        ("extract_letter_encoder", tc.ExtractLetterCategoricalEncoder("cabin")),
        (
            "categorical_inputer_encoder",
            tc.CategoricalImputerEncoder(variables=config.CATEGORICAL_VARS),
        ),
    ]
)

train_df = vars_without_persisting_info.transform(train_df)
test_df = vars_without_persisting_info.transform(test_df)

# Categorical variables with persisting information
rare_label_encoder = tc.RareLabelEncoder(tol=0.02, variables=["cabin"])
rare_label_encoder.fit(train_df)
train_df = rare_label_encoder.transform(train_df)
test_df = rare_label_encoder.transform(test_df)

one_hoter_encoder = tc.OneHotEncoderImputer(variables=config.CATEGORICAL_VARS)
train_df.drop(config.CATEGORICAL_VARS, axis=1, inplace=True)
test_df.drop(config.CATEGORICAL_VARS, axis=1, inplace=True)

# Numerical variables with persisting information
numerical_median_imputer = SimpleImputer(strategy="median")
numerical_median_imputer.fit(train_df[config.NUMERICAL_VARS])
train_df[config.NUMERICAL_VARS] = numerical_median_imputer.transform(
    train_df[config.NUMERICAL_VARS]
)
test_df[config.NUMERICAL_VARS] = numerical_median_imputer.transform(
    test_df[config.NUMERICAL_VARS]
)

min_max_scaler = MinMaxScaler()

# Numerical variables with persisting information - PIPELINE
numeric_pipeline = Pipeline(
    [("median_imputer", numerical_median_imputer), ("min_max_scaler", min_max_scaler)]
)
numeric_col_transformer = ColumnTransformer(
    [("numeric_transforms", numeric_pipeline, config.NUMERICAL_VARS)]
)

train_df[config.NUMERICAL_VARS] = numeric_col_transformer.fit_transform(train_df)
test_df[config.NUMERICAL_VARS] = numeric_col_transformer.fit_transform(test_df)

test_df = test_df[train_df.columns]

# Save processed DFs
train_df.to_csv(config.PROCESSED_TRAIN_DATA_FILE, index=False)
test_df.to_csv(config.PROCESSED_TEST_DATA_FILE, index=False)
