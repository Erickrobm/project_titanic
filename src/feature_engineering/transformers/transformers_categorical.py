import pandas as pd
import numpy as np
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from typing import List

# Transformes extra
class Cabin_Letter_Extractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        pass

    def transform(self, X: pd.DataFrame, y=None):
        X = pd.DataFrame(X).copy()
        X["cabin"] = X["cabin"].apply(
            lambda cabin: cabin[0] if type(cabin) == str else np.nan
        )
        return X


class GetTitle(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame):
        pass

    def transform(self, X: pd.DataFrame):
        X["title"] = [
            (name.split(",")[1]).split(".")[0].strip() for name in X["name"]
        ]
        return X


class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, variables: List[str] = None):
        self.variables = variables

    def fit(self, X: pd.DataFrame):
        pass

    def transform(self, X: pd.DataFrame):
        X = X.drop(self.variables, axis=1)
        return X


class ExtractLetterCategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, variables: str):
        self.variables = variables

    def fit(self, X: pd.DataFrame):
        pass

    def transform(self, X: pd.DataFrame):
        X[self.variables] = [
            "".join(re.findall("[a-zA-Z]+", x)) if type(x) == str else x
            for x in X[self.variables]
        ]
        return X


class CategoricalImputerEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, variables: List[str]):
        self.variables = variables

    def fit(self, X: pd.DataFrame):
        pass

    def transform(self, X: pd.DataFrame, variables: List[str]):
        X[self.variables] = X[self.variables].fillna("missing")
        return X


class RareLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, tol=0.02, variables: List[str] = None):
        self.tol = tol
        self.variables = variables

    def fit(self, X: pd.DataFrame, y=None):
        self.valid_labels_dict = {}
        for var in self.variables:
            t = X[var].value_counts() / X.shape[0]
            self.valid_labels_dict[var] = t[t > self.tol].index.tolist()

    def transform(self, X: pd.DataFrame, y=None):
        X = X.copy()
        for var in self.variables:
            tmp = [
                col for col in X[var].unique() if col not in self.valid_labels_dict[var]
            ]
            X[var] = X[var].replace(to_replace=tmp, value=len(tmp) * ["Rare"])
        return X


class OneHotEncoderImputer(BaseEstimator, TransformerMixin):
    def __init__(self, variables: List[str]):
        self.variables = variables

    def fit(self, X: pd.DataFrame):
        pass

    def transform(self, X: pd.DataFrame):
        enc = OneHotEncoder(handle_unknown="ignore", drop="first")
        enc.fit(X[self.variables])
        X[enc.get_feature_names_out(self.variables)] = enc.transform(
            X[self.variables]
        ).toarray()
        X = X.drop(self.variables, axis=1, inplace=True)
        return X