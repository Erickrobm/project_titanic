import pandas as pd
from typing import List
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer


class Missing_Indicator(BaseEstimator, TransformerMixin):
    """ """

    def __init__(self, variables: List[str]):
        self.variables = variables

    def fit(self, X: pd.DataFrame):
        """

        Parameters
        ----------
        X: pd.DataFrame : X_train or X_test dataframe.


        Returns
        -------

        """
        return self

    def transform(self, X: pd.DataFrame):
        """

        Parameters
        ----------
        X: X_train or X_test dataframe.


        Returns
        -------
        Returns _nan values in columns in database converted as int64.
        """
        for var in self.variables:
            X[var + "_nan"] = X[var].isnull().astype(int)
        return X


class MedianImputer(BaseEstimator, TransformerMixin):
    """ """

    def __init__(self, variables: List[str]):
        self.variables = variables

    def fit(self, X: pd.DataFrame, variables: List[str]):
        """

        Parameters
        ----------
        X: X_train or X_test dataframe.

        variables: List[str] : Numerical Variables Array.


        Returns
        -------
        Database filled with median values instead of nan values.
        """
        imp_median = SimpleImputer(strategy="median")
        imp_median.fit(X[self.variables])
        X[self.variables] = imp_median.transform(X[self.variables])
        return X

    def transform(self, X: pd.DataFrame):
        """

        Parameters
        ----------
        X: pd.DataFrame : X_train or X_test dataframe.


        Returns
        -------

        """
        return self
