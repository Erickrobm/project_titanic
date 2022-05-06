URL = "https://www.openml.org/data/get_csv/16826755/phpMYEkMl"
DROP_COLS = ["boat", "body", "home.dest", "ticket", "name"]
NUMERICAL_VARS = ["pclass", "age", "sibsp", "parch", "fare"]
CATEGORICAL_VARS = ["sex", "cabin", "embarked", "title"]
TARGET = "survived"
SEED_SPLIT = 404
SEED_MODEL = 404
TRAIN_DATA_FILE = 'data/cleaned/train.csv'
TEST_DATA_FILE = 'data/cleaned/test.csv'
#TRAIN_DATA_FILE = 'notebooks/train.csv'
#TEST_DATA_FILE = 'notebooks/test.csv'
TRAIN_TARGET_FILE = 'data/cleaned/y_train.csv'
TEST_TARGET_FILE = 'data/cleaned/y_test.csv'
PROCESSED_TRAIN_DATA_FILE = "data/processed/train.csv"
PROCESSED_TEST_DATA_FILE = "data/processed/test.csv"
PROCESSED_TRAIN_TARGET_FILE = "data/processed/y_train.csv"
PROCESSED_TEST_TARGET_FILE = "data/processed/y_test.csv"