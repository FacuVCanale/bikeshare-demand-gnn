import argparse
from pathlib import Path
import pandas as pd


def build_dataset(processed_dir: Path) -> pd.DataFrame:
    dispatch = pd.read_parquet(processed_dir / "dispatch_counts.parquet")
    arrival = pd.read_parquet(processed_dir / "arrival_counts.parquet")

    df = pd.merge(dispatch, arrival, on=["timestamp", "station_id"], how="outer").fillna(0)

    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    return df


def split_dataset(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    train = df[df["timestamp"] < "2024-07-01"]
    val = df[(df["timestamp"] >= "2024-07-01") & (df["timestamp"] < "2024-08-01")]
    test = df[df["timestamp"] >= "2024-08-01"]

    train.to_parquet(output_dir / "train.parquet")
    val.to_parquet(output_dir / "val.parquet")
    test.to_parquet(output_dir / "test.parquet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construcción de datasets para modelos")
    parser.add_argument("--processed_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    df = build_dataset(Path(args.processed_dir))
    split_dataset(df, Path(args.output_dir))
    print("Datasets guardados en", args.output_dir)
