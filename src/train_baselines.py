import argparse
from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_datasets(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(data_dir / "train.parquet")
    val = pd.read_parquet(data_dir / "val.parquet")
    test = pd.read_parquet(data_dir / "test.parquet")
    return train, val, test


def persistence_baseline(train: pd.DataFrame, eval_df: pd.DataFrame) -> pd.Series:
    df = pd.concat([train, eval_df]).sort_values("timestamp")
    df["pred"] = df.groupby("station_id")["arrival_count"].shift(1)
    return df.loc[eval_df.index, "pred"].fillna(0)


def mean_hour_baseline(train: pd.DataFrame, eval_df: pd.DataFrame) -> pd.Series:
    train = train.copy()
    train["hour_of_week"] = train["dayofweek"] * 24 + train["hour"]
    eval_df = eval_df.copy()
    eval_df["hour_of_week"] = eval_df["dayofweek"] * 24 + eval_df["hour"]
    mean = (
        train.groupby(["station_id", "hour_of_week"])["arrival_count"].mean().rename("pred")
    )
    joined = eval_df.set_index(["station_id", "hour_of_week"]).join(mean, how="left")
    return joined["pred"].fillna(0)


def evaluate(true: pd.Series, pred: pd.Series, name: str) -> None:
    mae = mean_absolute_error(true, pred)
    mse = mean_squared_error(true, pred)
    rmse = mse**0.5
    print(f"{name} - MAE: {mae:.3f} RMSE: {rmse:.3f}")


def main(data_dir: str) -> None:
    train, val, _ = load_datasets(Path(data_dir))

    pred_persistence = persistence_baseline(train, val)
    evaluate(val["arrival_count"], pred_persistence, "Persistencia")

    pred_mean = mean_hour_baseline(train, val)
    evaluate(val["arrival_count"], pred_mean, "Media por hora")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena baselines simples")
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()

    main(args.data_dir)
