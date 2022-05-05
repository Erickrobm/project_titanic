import pandas as pd
from utils import config as cf
from sklearn.linear_model import LogisticRegression
import joblib

X_train = pd.read_csv(cf.PROCESSED_TRAIN_DATA_FILE)
y_train = pd.read_csv(cf.TRAIN_TARGET_FILE).to_numpy().flatten()

logreg_model = LogisticRegression(
    C=0.0005, class_weight="balanced", random_state=cf.SEED_MODEL
)
logreg_model.fit(X_train, y_train)

joblib.dump(logreg_model, cf.MODEL_FILE)