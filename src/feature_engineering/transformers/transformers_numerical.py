import pandas as pd
from typing import List
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer


class Missing_Indicator(BaseEstimator, TransformerMixin):
    def __init__(self, variables:List[str]):
        self.variables= variables

    def fit(self, X:pd.DataFrame, variables:List[str]):
        self.variables= variables
        return self

    def transform(self, X: pd.DataFrame, variables:List[str]):
        self.variables=variables
        X = pd.DataFrame(X).copy()
        X[self.variables+"_nan"] = X[self.variables].isnull().astype(int)
        return X

class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, variables: List[str] = None):
        self.variables = variables

    def fit(self, X: pd.DataFrame):
        X = X.drop(self.variables, axis=1, inplace=True)
        return X

    def transform(self, X: pd.DataFrame):
        return self


class MedianImputer(BaseEstimator, TransformerMixin):
    def __init__(self, variables: List[str]):
        self.variables = variables

    def fit(self, X: pd.DataFrame, variables: List[str]):
        imp_median = SimpleImputer(strategy="median")
        imp_median.fit(X[self.variables])
        X[self.variables] = imp_median.transform(X[self.variables])
        return X

    def transform(self, X: pd.DataFrame):
        return self