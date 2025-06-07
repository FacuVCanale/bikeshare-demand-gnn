import argparse
from pathlib import Path
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_datasets(data_dir: Path):
    train = pd.read_parquet(data_dir / "train.parquet")
    val = pd.read_parquet(data_dir / "val.parquet")
    test = pd.read_parquet(data_dir / "test.parquet")
    return train, val, test


def prepare_features(df: pd.DataFrame):
    df = df.copy()
    df["station_id"] = df["station_id"].astype("category")
    cat = df["station_id"].cat.codes
    X = pd.DataFrame({
        "dispatch_count": df["dispatch_count"],
        "hour": df["hour"],
        "dayofweek": df["dayofweek"],
        "station": cat,
    })
    y = df["arrival_count"]
    return X, y


def evaluate(y_true, y_pred, name="Validation"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    print(f"{name} - MAE: {mae:.3f} RMSE: {rmse:.3f}")


def main(data_dir: str, models_dir: str):
    train, val, test = load_datasets(Path(data_dir))

    X_train, y_train = prepare_features(train)
    X_val, y_val = prepare_features(val)
    X_test, y_test = prepare_features(test)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": 6,
        "eta": 0.1,
    }

    model = xgb.train(params, dtrain, num_boost_round=500, evals=[(dval, "val")], early_stopping_rounds=20)

    Path(models_dir).mkdir(exist_ok=True, parents=True)
    model.save_model(str(Path(models_dir) / "xgb_model.json"))

    pred_val = model.predict(dval)
    evaluate(y_val, pred_val, name="Validation")

    dtest = xgb.DMatrix(X_test)
    pred_test = model.predict(dtest)
    evaluate(y_test, pred_test, name="Test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena modelo XGBoost")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--models_dir", type=str, default="models")
    args = parser.parse_args()

    main(args.data_dir, args.models_dir)
