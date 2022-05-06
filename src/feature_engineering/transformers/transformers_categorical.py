import pandas as pd
import numpy as np
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from typing import List

# Transformes extra
class Cabin_Letter_Extractor(BaseEstimator, TransformerMixin):
    """ """

    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        """

        Parameters
        ----------
        X: pd.DataFrame : URL Database.

        y : Not necessary to be declared.
             (Default value = None)

        Returns
        -------

        """
        pass

    def transform(self, X: pd.DataFrame, y=None):
        """

        Parameters
        ----------
        X: pd.DataFrame : URL Database.

        y : Not necessary to be declared.
             (Default value = None)

        Returns
        -------
        Database with only one cabin.
        """
        X = pd.DataFrame(X).copy()
        X["cabin"] = X["cabin"].apply(
            lambda cabin: cabin[0] if type(cabin) == str else np.nan
        )
        return X


class GetTitle(BaseEstimator, TransformerMixin):
    """ """

    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame):
        """

        Parameters
        ----------
        X: pd.DataFrame : URL Database.

        Returns
        -------

        """
        pass

    def transform(self, X: pd.DataFrame):
        """

        Parameters
        ----------
        X: pd.DataFrame : URL Dataframe.


        Returns
        -------
        Database with names extracted, only kept sufixes like "Mrs, Mr,
        Miss and Master".
        """
        X["title"] = [(name.split(",")[1]).split(".")[0].strip() for name in X["name"]]
        return X


class DropColumns(BaseEstimator, TransformerMixin):
    """ """

    def __init__(self, X: pd.DataFrame, variables: List[str]):
        self.X = X
        self.variables = variables

    def fit(self, X: pd.DataFrame):
        """

        Parameters
        ----------
        X: pd.DataFrame : URL Dataframe.

        variables: List[str] : Dropped Columns Array.

        Returns
        -------

        """
        pass

    def transform(self, X: pd.DataFrame):
        """

        Parameters
        ----------
        X: pd.DataFrame : Dropped Columns Array.


        Returns
        -------
        Database with irrelevant columns removed, like 'boat','body',
        'home.dest','ticket' and 'name'.
        """
        X = X.drop(self.variables, axis=1)
        return X


class ExtractLetterCategoricalEncoder(BaseEstimator, TransformerMixin):
    """ """

    def __init__(self, variables: str):
        self.variables = variables

    def fit(self, X: pd.DataFrame):
        """

        Parameters
        ----------
        X: pd.DataFrame :


        Returns
        -------

        """
        pass

    def transform(self, X: pd.DataFrame):
        """

        Parameters
        ----------
        X: pd.DataFrame : Must be the 'cabin' variable [].


        Returns
        -------
        Returns the letters on the cabin column.
        """
        X[self.variables] = [
            "".join(re.findall("[a-zA-Z]+", x)) if type(x) == str else x
            for x in X[self.variables]
        ]
        return X


class CategoricalImputerEncoder(BaseEstimator, TransformerMixin):
    """ """

    def __init__(self, variables: List[str]):
        self.variables = variables

    def fit(self, X: pd.DataFrame):
        """

        Parameters
        ----------
        X: pd.DataFrame :


        Returns
        -------

        """
        return self

    def transform(self, X: pd.DataFrame):
        """

        Parameters
        ----------
        X: pd.DataFrame : Categorical Variables Array.


        Returns
        -------
        Returns the X_train/X_test dataset with extra numerical variables columns
        indicating the number of _nan values on each.
        """
        X[self.variables] = X[self.variables].fillna("missing")
        return X


class RareLabelEncoder(BaseEstimator, TransformerMixin):
    """ """

    def __init__(self, tol=0.02, variables: List[str] = None):
        self.tol = tol
        self.variables = variables

    def fit(self, X: pd.DataFrame, y=None):
        """

        Parameters
        ----------
        X: Must be an specific variable inside the Categorical Variables
        Array e.g. ['cabin'].

        y :
             (Default value = None)

        Returns
        -------
        Returns the exact number of missing and rare data in cloumns.
        """
        self.valid_labels_dict = {}
        for var in self.variables:
            t = X[var].value_counts() / X.shape[0]
            self.valid_labels_dict[var] = t[t > self.tol].index.tolist()

    def transform(self, X: pd.DataFrame, y=None):
        """

        Parameters
        ----------
        X: pd.DataFrame :

        y :
             (Default value = None)

        Returns
        -------

        """
        X = X.copy()
        for var in self.variables:
            tmp = [
                col for col in X[var].unique() if col not in self.valid_labels_dict[var]
            ]
            X[var] = X[var].replace(to_replace=tmp, value=len(tmp) * ["Rare"])
        return X


class OneHotEncoderImputer(BaseEstimator, TransformerMixin):
    """ """

    def __init__(self, variables: List[str]):
        self.variables = variables

    def fit(self, X: pd.DataFrame):
        """

        Parameters
        ----------
        X: pd.DataFrame :


        Returns
        -------

        """
        pass

    def transform(self, X: pd.DataFrame):
        """

        Parameters
        ----------
        X: pd.DataFrame : Categorical Variables Array.


        Returns
        -------
        Returns database with "object" columns removed, only remained float64
        and int64 columns.
        """
        enc = OneHotEncoder(handle_unknown="ignore", drop="first")
        enc.fit(X[self.variables])
        X[enc.get_feature_names_out(self.variables)] = enc.transform(
            X[self.variables]
        ).toarray()
        X = X.drop(self.variables, axis=1, inplace=True)
        return X
