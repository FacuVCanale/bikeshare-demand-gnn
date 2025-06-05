import argparse
from pathlib import Path
import pandas as pd


def load_trips(input_dir: Path) -> pd.DataFrame:
    """Carga todos los CSV de viajes en un solo DataFrame."""
    files = sorted(input_dir.glob('*.csv'))
    dfs = []
    for f in files:
        df = pd.read_csv(
            f,
            parse_dates=["fecha_origen", "fecha_destino"],
            dayfirst=True,
        )
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError("No se encontraron CSV en " + str(input_dir))
    return pd.concat(dfs, ignore_index=True)


def compute_counts(df: pd.DataFrame, dt_minutes: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve dos DataFrames con conteos de partidas y arribos por estación."""
    df = df.copy()
    df["window_dispatch"] = df["fecha_origen"].dt.floor(f"{dt_minutes}min")
    df["window_arrival"] = df["fecha_destino"].dt.floor(f"{dt_minutes}min")

    dispatch = (
        df.groupby(["window_dispatch", "estacion_origen_id"])  # type: ignore
        .size()
        .reset_index(name="dispatch_count")
        .rename(columns={"window_dispatch": "timestamp", "estacion_origen_id": "station_id"})
    )

    arrival = (
        df.groupby(["window_arrival", "estacion_destino_id"])  # type: ignore
        .size()
        .reset_index(name="arrival_count")
        .rename(columns={"window_arrival": "timestamp", "estacion_destino_id": "station_id"})
    )

    return dispatch, arrival


def main(input_dir: str, output_dir: str, dt_minutes: int = 30) -> None:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    trips = load_trips(input_path)
    dispatch, arrival = compute_counts(trips, dt_minutes=dt_minutes)

    dispatch.to_parquet(output_path / "dispatch_counts.parquet")
    arrival.to_parquet(output_path / "arrival_counts.parquet")
    print("Archivos guardados en", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de procesamiento de EcoBici")
    parser.add_argument("--input_dir", type=str, required=True, help="Directorio con CSV crudos")
    parser.add_argument("--output_dir", type=str, required=True, help="Directorio destino para parquet")
    parser.add_argument("--dt_minutes", type=int, default=30, help="Ventana de tiempo en minutos")
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, dt_minutes=args.dt_minutes)
