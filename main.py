import pandas as pd
from data.utils import config as cf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from src.pipeline import titanic_pipeline

df = pd.read_csv(cf.URL, na_values="?")

df.to_csv("data/raw/raw_titanic.csv")

X_train, X_test, y_train, y_test = train_test_split(
    df.drop(cf.TARGET, axis=1), df[cf.TARGET], test_size=0.2, random_state=cf.SEED_SPLIT
)

X_train.to_csv(cf.TRAIN_DATA_FILE, index=False)
X_test.to_csv(cf.TEST_DATA_FILE, index=False)

titanic_pipeline.fit(X_train, y_test)

class_pred = titanic_pipeline.predict(X_test)
proba_pred = titanic_pipeline.predict_proba(X_test)[:, 1]
print("test roc-auc : {}".format(roc_auc_score(y_test, proba_pred)))
print("test accuracy: {}".format(accuracy_score(y_test, class_pred)))
