from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

PROCESSED = Path("data/processed/train_ready.csv")

CATEGORICAL = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
]

NUMERIC = [
    "age",
    "campaign",
    "previous",
    "was_contacted_before",
    "pdays_clean",
    "emp_var_rate",
    "cons_price_idx",
    "cons_conf_idx",
    "euribor3m",
    "nr_employed",
]

FEATURES = CATEGORICAL + NUMERIC
TARGET = "y"
RANDOM_STATE = 42


def load_processed() -> pd.DataFrame:
    return pd.read_csv(PROCESSED)


def split(df: pd.DataFrame):
    X = df[FEATURES + ["duration"]].copy()
    y = df[TARGET].copy()
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_STATE
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def drop_duration(X: pd.DataFrame) -> pd.DataFrame:
    return X.drop(columns=["duration"])
