import pandas as pd
from sklearn.model_selection import train_test_split
from data.cleaned.utils import config

# Loading data from specific url
df = pd.read_csv(config.URL, na_values="?")

df["cabin"] = df["cabin"].apply(lambda x: x[0] if type(x) == str else x)
df["title"] = (
    df["name"]
    .apply(lambda x: (x.split(",")[1]).split(".")[0])
    .apply(
        lambda x: x
        if (x.lower()).strip() in ["mrs", "mr", "miss", "master"]
        else "other"
    )
)
df.drop(columns=config.DROP_COLS, inplace=True)

# Write data
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(config.TARGET, axis=1),
    df[config.TARGET],
    test_size=0.2,
    random_state=config.SEED_MODEL,
)

X_train.to_csv(config.TRAIN_DATA_FILE, index=False)
X_test.to_csv(config.TEST_DATA_FILE, index=False)

y_train.to_csv(config.TRAIN_TARGET_FILE, index=False)
y_test.to_csv(config.TEST_TARGET_FILE, index=False)