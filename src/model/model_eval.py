import pandas as pd
import joblib
from utils import config as cf
from sklearn.metrics import accuracy_score, roc_auc_score


X_train = pd.read_csv(cf.PROCESSED_TRAIN_DATA_FILE)
y_train = pd.read_csv(cf.TRAIN_TARGET_FILE)
X_test = pd.read_csv(cf.PROCESSED_TEST_DATA_FILE)
y_test = pd.read_csv(cf.TEST_TARGET_FILE)

log_reg_model = joblib.load(cf.MODEL_FILE)

for s, Xy in zip(["train", "test"], [(X_train, y_train), (X_test, y_test)]):
    print(f"\n")
    x, y = Xy[0], Xy[1]
    class_pred = log_reg_model.predict(x)
    proba_pred = log_reg_model.predict_proba(x)[:, 1]
    print("{} roc-auc : {}".format(s, roc_auc_score(y, proba_pred)))
    print("{} accuracy: {}".format(s, accuracy_score(y, class_pred)))
print(f"\n")