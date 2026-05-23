from pathlib import Path
import pandas as pd

RAW = Path("data/raw/bank_additional_clean.csv")
OUT = Path("data/processed/train_ready.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW)
    return df


def fill_unknowns(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = df.select_dtypes(include="object").columns
    df[cat_cols] = df[cat_cols].fillna("unknown")
    return df


def engineer_pdays(df: pd.DataFrame) -> pd.DataFrame:
    df["was_contacted_before"] = (df["pdays"] != 999).astype(int)
    df["pdays_clean"] = df["pdays"].where(df["pdays"] != 999)
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    df["y"] = (df["y"] == "yes").astype(int)
    return df


def main() -> None:
    df = load_raw()
    print(f"raw shape: {df.shape}")
    df = fill_unknowns(df)
    df = engineer_pdays(df)
    df = encode_target(df)
    print(f"target rate: {df['y'].mean():.4f}")
    print(f"contacted-before rate: {df['was_contacted_before'].mean():.4f}")
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
