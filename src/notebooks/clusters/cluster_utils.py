import polars as pl
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import skellam
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import joblib

import os
import gc
from pathlib import Path

def write_parquet_streaming(
    df: pl.DataFrame,
    path: str | Path,
    *,
    compression: str = "snappy",
    row_group_size: int = 10_000,
) -> None:
    """
    Persist a large Polars DataFrame to Parquet using the built-in streaming engine.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame to write.
    path : str | pathlib.Path
        Destination file path.
    compression : {'snappy', 'zstd', 'gzip', 'lz4', 'uncompressed'}, default 'snappy'
        Parquet compression codec.
    row_group_size : int, default 10_000
        Row-group size (balance between seek-time and file size).

    Notes
    -----
    * The function relies on Polars' streaming sink, which writes the data
    incrementally without reading it back or mmap-locking the file.
    * Works best with Polars ≥ 1.0, but the `engine="streaming"` flag is added
    explicitly for clarity.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Lazy frame → streaming sink
    (
        df.lazy()
        .sink_parquet(
            path.as_posix(),
            compression=compression,
            row_group_size=row_group_size,
            engine="streaming",          # explicit; Polars picks streaming automatically on large frames
        )
    )

    print(f"✅ Saved {df.height:,} rows to {path.resolve()}")

def merge_intra_clusters(df: pl.DataFrame) -> pl.DataFrame:
    """
    Merges rows within the same cluster and timestamp by summing arrivals and departures columns.
    
    For each unique combination of (ts_start, cluster), combines all stations' data into a single row.
    This aggregates the bike sharing activity across all stations within a cluster for each time period.
    
    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with columns including ts_start, cluster, and various arrival/departure lag columns
        
    Returns
    -------
    pl.DataFrame
        Aggregated DataFrame with one row per (ts_start, cluster) combination
    """
    # Define columns to aggregate (sum)
    agg_columns = [
        'dep_last_DT', 'trip_dur_mean_last_DT',
        'dep_lag_1', 'dep_lag_2', 'dep_lag_3', 'dep_lag_4', 'dep_lag_5', 'dep_lag_6',
        'arr_last_DT', 'arr_lag_1', 'arr_lag_2', 'arr_lag_3', 'arr_lag_4', 'arr_lag_5', 'arr_lag_6',
        # Add demand lags
        'dem_lag_1', 'dem_lag_2', 'dem_lag_3', 'dem_lag_4', 'dem_lag_5', 'dem_lag_6',
        'y_arrivals_next_DT', 'y_departures_next_DT'
    ]
    
    # Keep only existing columns
    agg_columns = [col for col in agg_columns if col in df.columns]
    
    # Group by cluster and timestamp, then sum the relevant columns
    groupby_cols = ['ts_start', 'cluster']
    
    # Create aggregation expressions for summing
    sum_exprs = [pl.col(col).sum() for col in agg_columns if col != 'trip_dur_mean_last_DT']
    
    # Add trip_dur_mean_last_DT as mean if it exists
    if 'trip_dur_mean_last_DT' in agg_columns:
        sum_exprs.append(pl.col('trip_dur_mean_last_DT').mean())
    
    # Add mean of lat and lon as new features
    if 'lat' in df.columns:
        sum_exprs.append(pl.col('lat').mean().alias('lat'))
    if 'lon' in df.columns:
        sum_exprs.append(pl.col('lon').mean().alias('lon'))
    
    # Add other columns as first value (they should be the same for all stations in the same cluster and timestamp)
    other_columns = [col for col in df.columns if col not in agg_columns and col not in groupby_cols and col not in ['lat', 'lon']]
    for col in other_columns:
        sum_exprs.append(pl.col(col).first().alias(col))
    
    # Perform aggregation
    df_merged = df.group_by(groupby_cols).agg(sum_exprs)
    
    # Recalculate delta_t if the target columns exist
    if 'y_arrivals_next_DT' in df_merged.columns and 'y_departures_next_DT' in df_merged.columns:
        df_merged = df_merged.with_columns(
            (pl.col('y_arrivals_next_DT') - pl.col('y_departures_next_DT')).alias('delta_t')
        )
    
    # Create demand lag columns if arrivals and departures exist
    if 'arr_lag_1' in df_merged.columns and 'dep_lag_1' in df_merged.columns:
        for i in range(1, 7):  # lags 1-6
            arr_col = f'arr_lag_{i}'
            dep_col = f'dep_lag_{i}'
            dem_col = f'dem_lag_{i}'
            if arr_col in df_merged.columns and dep_col in df_merged.columns:
                df_merged = df_merged.with_columns(
                    (pl.col(arr_col) - pl.col(dep_col)).alias(dem_col)
                )
    
    print(f"Merged {df.height:,} station-level rows into {df_merged.height:,} cluster-level rows")
    print(f"Reduction factor: {df.height / df_merged.height:.1f}x")
    
    return df_merged

def dataset_to_cluster_train(dataset_path: str, output_path: str, sample=5_000_000, seed=42, cluster_range=range(41, 100)):

    # Set the random seed for reproducibility
    np.random.seed(seed)

    df = pl.read_parquet(dataset_path)
    null_counts = df.null_count().to_dict(as_series=False)
    cols_to_fill = [col for col in df.columns if (not col.startswith('weather_')) and (null_counts[col][0] > 0)]
    df = df.with_columns([pl.col(col).fill_null(0) for col in cols_to_fill])

    # Sample directly from Polars (much more memory efficient)
    n_sample = min(sample, df.shape[0])
    sample_fraction = n_sample / df.shape[0]

    # Use Polars random sampling (memory efficient)
    df_sample = df.sample(fraction=sample_fraction, seed=seed)
    print(f"Sampled dataset size: {df_sample.shape[0]:,} rows ({sample_fraction*100:.1f}%)")

    # Now convert only the sample to pandas
    pdf = df_sample.to_pandas()
    print(f"Memory usage after sampling: ~{pdf.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    # Clear original large dataframes
    del df, df_sample

    # Define targets: Δ_t = arrivals - departures 
    target_A = 'y_arrivals_next_DT'
    target_D = 'y_departures_next_DT'

    for col in [target_A, target_D]:
        pdf[col] = pdf[col].astype(np.int64)      # o 'Int64' de pandas para nullables

    pdf['delta_t'] = pdf[target_A] - pdf[target_D]

    print("Extracting unique stations...")
    unique_stations = pdf[['station_id', 'lat', 'lon']].drop_duplicates('station_id')
    print(f"Found {len(unique_stations)} unique stations")

    # 2. Prepare coordinates for clustering
    coords = unique_stations[['lat', 'lon']].values
    print(f"Coordinate range: lat=[{coords[:, 0].min():.4f}, {coords[:, 0].max():.4f}], lon=[{coords[:, 1].min():.4f}, {coords[:, 1].max():.4f}]")

    # Scale coordinates for GMM
    scaler_geo = StandardScaler()
    coords_scaled = scaler_geo.fit_transform(coords)

    # 3. Find optimal number of clusters using silhouette score
    silhouette_scores = []
    models = {}

    print("Testing different numbers of clusters...")
    for n_clusters in cluster_range:
        # Fit GMM
        gmm = GaussianMixture(n_components=n_clusters, random_state=42, n_init=5)
        cluster_labels = gmm.fit_predict(coords_scaled)
        
        # Calculate silhouette score
        sil_score = silhouette_score(coords_scaled, cluster_labels)
        silhouette_scores.append(sil_score)
        models[n_clusters] = gmm
        
        print(f"n_clusters={n_clusters:2d}: silhouette_score={sil_score:.4f}")

    # Find optimal number of clusters
    optimal_n_clusters = cluster_range[np.argmax(silhouette_scores)]
    print(f"\nOptimal number of clusters: {optimal_n_clusters}")
    print(f"Best silhouette score: {max(silhouette_scores):.4f}")

    # Get the optimal model and final cluster assignments
    best_gmm = models[optimal_n_clusters]
    final_clusters = best_gmm.predict(coords_scaled)
    unique_stations['cluster'] = final_clusters

    # Merge cluster assignments back to the main dataset
    pdf_clustered = pdf.merge(unique_stations[['station_id', 'cluster']], on='station_id', how='left')

    # Verify the merge
    print(f"Original dataset shape: {pdf.shape}")
    print(f"Clustered dataset shape: {pdf_clustered.shape}")
    print(f"Samples with cluster labels: {pdf_clustered['cluster'].notna().sum()}/{len(pdf_clustered)}")


    output_path = Path(output_path)
    if output_path.parent != Path('.'):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert back to Polars for efficient chunked saving
    print("Converting to Polars for chunked saving...")
    df_clustered_pl = pl.from_pandas(pdf_clustered)

    # Save as parquet using chunked writing
    print(f"Saving clustered dataset to: {output_path}")
    write_parquet_streaming(df_clustered_pl, str(output_path), row_group_size=100000)
    print(f"File size: ~{os.path.getsize(output_path) / 1e6:.1f} MB")

    # Clear memory
    del df_clustered_pl
    gc.collect()

    # Also save the station-cluster mapping for future use
    mapping_file = os.path.join(output_path.parent, "station_cluster_mapping.csv")
    unique_stations.to_csv(mapping_file, index=False)
    print(f"Saved station-cluster mapping to: {mapping_file}")

    # Save clustering model for future predictions
    model_file = os.path.join(output_path.parent, "geographic_clustering_model.pkl")
    joblib.dump({
        'gmm_model': best_gmm,
        'scaler': scaler_geo,
        'n_clusters': optimal_n_clusters,
        'silhouette_score': max(silhouette_scores)
    }, model_file)
    print(f"Saved clustering model to: {model_file}")

    return pdf_clustered

# Apply targeted outlier removal per cluster for trip_dur_mean_last_DT
def remove_top_outliers_per_cluster(df, column, top_n=5):
    """
    Remove only the top N extreme outliers from a specific column within each cluster.
    
    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame
    column : str
        Column to check for outliers
    top_n : int, default 5
        Number of top extreme values to remove per cluster
    
    Returns
    -------
    pl.DataFrame
        DataFrame with top outliers removed per cluster
    """
    print(f"🔍 Removing top {top_n} outliers per cluster from column: {column}")
    print(f"Original data shape: {df.shape}")
    
    if column not in df.columns:
        print(f"Column {column} not found in DataFrame")
        return df
    
    if 'cluster' not in df.columns:
        print(f"Column 'cluster' not found in DataFrame")
        return df
    
    # Get unique clusters
    clusters = df['cluster'].unique().to_list()
    print(f"Processing {len(clusters)} clusters")
    
    # Process each cluster separately
    clean_dfs = []
    total_removed = 0
    
    for cluster_id in clusters:
        cluster_df = df.filter(pl.col('cluster') == cluster_id)
        
        if cluster_df.height <= top_n:
            # If cluster has fewer rows than top_n, keep all rows
            print(f"  Cluster {cluster_id}: {cluster_df.height} rows, keeping all")
            clean_dfs.append(cluster_df)
            continue
        
        # Get the top N highest values to remove for this cluster
        top_values = cluster_df[column].top_k(top_n)
        
        # Create mask to exclude rows with these top values
        mask = pl.lit(True)
        for val in top_values.to_list():
            if val is not None:
                mask = mask & (pl.col(column) != val)
        
        cluster_clean = cluster_df.filter(mask)
        
        n_removed_cluster = cluster_df.height - cluster_clean.height
        total_removed += n_removed_cluster
        
        print(f"  Cluster {cluster_id}: removed {n_removed_cluster} rows (top values: {top_values.to_list()})")
        clean_dfs.append(cluster_clean)
    
    # Concatenate all clean cluster dataframes
    df_clean = pl.concat(clean_dfs)
    
    print(f"📊 Total removed {total_removed} rows ({total_removed/df.shape[0]*100:.2f}%) with top outliers across all clusters")
    print(f"Clean data shape: {df_clean.shape}")
    
    return df_clean


def train_cluster_baseline(cluster_data, cluster_id, feature_cols=None, target_A=None, target_D=None, n_splits=3):
    """
    Train dual Poisson regression baseline for a single cluster
    Returns aggregated results across CV folds
    """
    try:
        # Prepare features and targets
        X = cluster_data[feature_cols].copy()
        # Convert all features to float64
        X = X.astype(np.float64)
        y_A = cluster_data[target_A].values
        y_D = cluster_data[target_D].values

        # 🔍 DIAGNOSTIC: Check target ranges BEFORE processing
        print(f"    Raw targets - A: min={y_A.min()}, max={y_A.max()}, mean={y_A.mean():.3f}")
        print(f"    Raw targets - D: min={y_D.min()}, max={y_D.max()}, mean={y_D.mean():.3f}")

        # Handle any remaining NaN values
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Ensure non-negative targets and proper data types
        y_A = np.maximum(y_A, 0).astype(np.float64)
        y_D = np.maximum(y_D, 0).astype(np.float64)

        # 🔍 DIAGNOSTIC: Check target ranges AFTER processing
        print(f"    Clean targets - A: min={y_A.min()}, max={y_A.max()}, mean={y_A.mean():.3f}")
        print(f"    Clean targets - D: min={y_D.min()}, max={y_D.max()}, mean={y_D.mean():.3f}")

        # Check for suspicious values that could cause huge errors
        if y_A.max() > 10000 or y_D.max() > 10000:
            print(f"    ⚠️  WARNING: Extremely high target values detected!")
            print(f"    This may be causing the huge MAE/RMSE values!")
            return {
                'cluster_id': cluster_id,
                'n_samples': len(cluster_data),
                'mae_mean': np.nan,
                'convergence_success': False,
                'error': f'Suspicious target values: A_max={y_A.max()}, D_max={y_D.max()}'
            }

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_results = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_A_train, y_A_test = y_A[train_idx], y_A[test_idx]
            y_D_train, y_D_test = y_D[train_idx], y_D[test_idx]

            model_A = PoissonRegressor(alpha=1e-4, max_iter=1000, tol=1e-4)
            model_D = PoissonRegressor(alpha=1e-4, max_iter=1000, tol=1e-4)
            model_A.fit(X_train, y_A_train)
            model_D.fit(X_train, y_D_train)

            lambda_A = model_A.predict(X_test)
            lambda_D = model_D.predict(X_test)

            # Check for extreme predictions
            for i, pred in enumerate(lambda_A):
                if pred > 100:
                    print(f"  Fold {fold} - Lambda_A > 100: {pred:.3e}")

            for i, pred in enumerate(lambda_D):
                if pred > 100:
                    print(f"  Fold {fold} - Lambda_D > 100: {pred:.3e}")
            
            # Combined predictions
            point_forecast = lambda_A - lambda_D
            var_forecast = lambda_A + lambda_D
            y_delta_test = y_A_test - y_D_test

            # Filter out ridiculous predictions
            lam_mask = (lambda_A > 40) | (lambda_D > 40)
            keep_idx = ~lam_mask

            n_out = np.sum(~keep_idx)
            n_keep = np.sum(keep_idx)
            if n_keep == 0:
                print(f"      Fold {fold} – skipped (only outliers detected, {n_out} rows)")
                continue
            if n_out > 0:
                print(f"      Fold {fold} – dropped {n_out} outlier rows")

            lambda_A = lambda_A[keep_idx]
            lambda_D = lambda_D[keep_idx]
            point_forecast = point_forecast[keep_idx]
            y_delta_test = y_delta_test[keep_idx]

            # 🔍 DIAGNOSTIC: Check prediction ranges
            print(f"      Fold {fold} - Lambda_A: {lambda_A.min():.3f} to {lambda_A.max():.3f} (mean={lambda_A.mean():.3f})")
            print(f"      Fold {fold} - Lambda_D: {lambda_D.min():.3f} to {lambda_D.max():.3f} (mean={lambda_D.mean():.3f})")
            print(f"      Fold {fold} - y_delta_test: {y_delta_test.min():.1f} to {y_delta_test.max():.1f} (mean={y_delta_test.mean():.3f})")
            print(f"      Fold {fold} - point_forecast: {point_forecast.min():.3f} to {point_forecast.max():.3f} (mean={point_forecast.mean():.3f})")

            # Metrics
            mae = mean_absolute_error(y_delta_test, point_forecast)
            rmse = np.sqrt(mean_squared_error(y_delta_test, point_forecast))
            r2 = r2_score(y_delta_test, point_forecast) if len(np.unique(y_delta_test)) > 1 else -999

            print(f"      Fold {fold} - MAE: {mae:.6f}, RMSE: {rmse:.6f}, R2: {r2:.6f}")

            # Check for problematic values
            if mae > 1000 or rmse > 1000:
                print(f"      ⚠️  EXTREME VALUES DETECTED!")
                print(f"      This suggests data corruption or scale issues!")
                return {
                    'cluster_id': cluster_id,
                    'n_samples': len(cluster_data),
                    'mae_mean': np.nan,
                    'convergence_success': False,
                    'error': f'Extreme MAE/RMSE values in fold {fold}: MAE={mae:.1f}, RMSE={rmse:.1f}'
                }

            # Skellam log-likelihood (handle potential numerical issues)
            try:
                ll = np.sum(skellam.logpmf(y_delta_test.astype(int), lambda_A, lambda_D))
                if np.isnan(ll) or np.isinf(ll):
                    ll = -999999
            except:
                ll = -999999

            # PIT calibration
            try:
                pit_values = skellam.cdf(y_delta_test.astype(int), lambda_A, lambda_D)
                pit_ks = np.abs(np.sort(pit_values) - np.linspace(0, 1, len(pit_values))).max()
            except:
                pit_ks = 1.0  # Worst case

            fold_results.append({
                'fold': fold,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'log_likelihood': ll,
                'pit_uniform_ks': pit_ks,
                'lambda_A_mean': lambda_A.mean(),
                'lambda_D_mean': lambda_D.mean(),
                'var_forecast_mean': var_forecast.mean(),
                'n_test': len(y_delta_test)
            })

        # Aggregate results across folds
        cluster_result = {
            'cluster_id': cluster_id,
            'n_samples': len(cluster_data),
            'n_folds': len(fold_results),
            'mae_mean': np.mean([r['mae'] for r in fold_results]),
            'mae_std': np.std([r['mae'] for r in fold_results]),
            'rmse_mean': np.mean([r['rmse'] for r in fold_results]),
            'rmse_std': np.std([r['rmse'] for r in fold_results]),
            'r2_mean': np.mean([r['r2'] for r in fold_results if r['r2'] != -999]),
            'r2_std': np.std([r['r2'] for r in fold_results if r['r2'] != -999]),
            'log_likelihood_mean': np.mean([r['log_likelihood'] for r in fold_results if r['log_likelihood'] != -999999]),
            'pit_ks_mean': np.mean([r['pit_uniform_ks'] for r in fold_results]),
            'lambda_A_mean': np.mean([r['lambda_A_mean'] for r in fold_results]),
            'lambda_D_mean': np.mean([r['lambda_D_mean'] for r in fold_results]),
            'convergence_success': True
        }

        return cluster_result

    except Exception as e:
        print(f"Error training cluster {cluster_id}: {str(e)}")
        return {
            'cluster_id': cluster_id,
            'n_samples': len(cluster_data),
            'mae_mean': np.nan,
            'convergence_success': False,
            'error': str(e)
        }

def train_cluster_baseline_v2(
    cluster_data,
    cluster_id,
    feature_cols,
    target_A,
    target_D,
    *,
    n_splits=3,
    LAMBDA_MAX=40.0,
    poisson_alpha=1e-4,
    poisson_max_iter=1000,
    poisson_tol=1e-4,
    scaler_cls=StandardScaler,
    fillna_strategy="median"
):
    """
    Baseline de Poisson doble por cluster, sin fugas de datos y con trazas de outliers.
    Imprime filas con λA o λD > LAMBDA_MAX.
    Devuelve dict con métricas agregadas.

    Parameters
    ----------
    cluster_data : pd.DataFrame
        Data for a single cluster.
    cluster_id : int or str
        Cluster identifier.
    feature_cols : list of str
        Feature columns to use.
    target_A : str
        Name of arrivals target column.
    target_D : str
        Name of departures target column.
    n_splits : int, default 3
        Number of splits for TimeSeriesSplit.
    LAMBDA_MAX : float, default 40.0
        Maximum allowed lambda for outlier detection.
    poisson_alpha : float, default 1e-4
        Regularization strength for PoissonRegressor.
    poisson_max_iter : int, default 1000
        Maximum iterations for PoissonRegressor.
    poisson_tol : float, default 1e-4
        Tolerance for PoissonRegressor.
    scaler_cls : class, default StandardScaler
        Scaler class to use for feature normalization.
    fillna_strategy : str, default "median"
        Strategy for filling NaNs in features ("median" or "mean").
    """
    try:
        # -------- 0. Orden temporal (evita fugas en CV) ----------
        cluster_data = cluster_data.sort_values("ts_start").reset_index(drop=True)

        if fillna_strategy == "median":
            fill_value = cluster_data[feature_cols].median()
        elif fillna_strategy == "mean":
            fill_value = cluster_data[feature_cols].mean()
        else:
            raise ValueError(f"Unknown fillna_strategy: {fillna_strategy}")

        X_full = (
            cluster_data[feature_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(fill_value)
            .astype(np.float64)
        )

        # Additional safety check for any remaining NaNs
        if X_full.isna().any().any():
            print(f"⚠️  Warning: Still have NaNs after fillna, filling with 0")
            X_full = X_full.fillna(0)

        y_A_full = np.maximum(cluster_data[target_A].values, 0).astype(np.float64)
        y_D_full = np.maximum(cluster_data[target_D].values, 0).astype(np.float64)

        print(f"[{cluster_id}]  muestras = {len(cluster_data):,}")

        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_results = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_full)):
            # -- 1. Escalador SOLO con datos de entrenamiento
            scaler = scaler_cls()
            X_train = scaler.fit_transform(X_full.iloc[train_idx])
            X_test  = scaler.transform(X_full.iloc[test_idx])

            yA_tr, yA_ts = y_A_full[train_idx], y_A_full[test_idx]
            yD_tr, yD_ts = y_D_full[train_idx], y_D_full[test_idx]

            # -- 2. Entrenamiento
            model_A = PoissonRegressor(alpha=poisson_alpha, max_iter=poisson_max_iter, tol=poisson_tol)
            model_D = PoissonRegressor(alpha=poisson_alpha, max_iter=poisson_max_iter, tol=poisson_tol)
            model_A.fit(X_train, yA_tr)
            model_D.fit(X_train, yD_tr)

            λA = model_A.predict(X_test)
            λD = model_D.predict(X_test)

            # -- 3. Detectar y mostrar outliers
            bad_mask = (λA > LAMBDA_MAX) | (λD > LAMBDA_MAX)
            if bad_mask.any():
                for i_local, flag in enumerate(bad_mask):
                    if flag:
                        global_i = test_idx[i_local]
                        print(f"🚨 Fold {fold} – λA={λA[i_local]:.2f}, λD={λD[i_local]:.2f}")
                        # Show normalized features that are extreme outliers (>3 std devs)
                        normalized_features = X_test[i_local]
                        extreme_mask = np.abs(normalized_features) > 3
                        if extreme_mask.any():
                            extreme_cols = np.array(feature_cols)[extreme_mask]
                            extreme_vals = normalized_features[extreme_mask]
                            print(f"Extreme normalized features (>3σ): {dict(zip(extreme_cols, extreme_vals))}")
                        else:
                            print("No extreme normalized features (>3σ)")
                # excluimos esos puntos
            keep_mask = ~bad_mask
            if keep_mask.sum() == 0:
                print(f"⚠️  Fold {fold} descartada: solo outliers")
                continue

            λA, λD      = λA[keep_mask], λD[keep_mask]
            y_delta_ts  = (yA_ts - yD_ts)[keep_mask]
            point_fcst  = λA - λD
            var_fcst    = λA + λD  # ya filtrado

            # -- 4. Métricas
            mae  = mean_absolute_error(y_delta_ts, point_fcst)
            rmse = np.sqrt(mean_squared_error(y_delta_ts, point_fcst))
            r2   = r2_score(y_delta_ts, point_fcst) if len(np.unique(y_delta_ts))>1 else np.nan

            try:
                ll = skellam.logpmf(y_delta_ts.astype(int), λA, λD).sum()
            except Exception:
                ll = np.nan

            try:
                pit_ks = np.abs(
                    np.sort(skellam.cdf(y_delta_ts.astype(int), λA, λD))
                    - np.linspace(0, 1, len(y_delta_ts))
                ).max()
            except Exception:
                pit_ks = np.nan

            fold_results.append(dict(
                fold=fold, mae=mae, rmse=rmse, r2=r2, ll=ll,
                pit_ks=pit_ks, lambda_A_mean=λA.mean(), lambda_D_mean=λD.mean(),
                var_forecast_mean=var_fcst.mean(), n_test=len(y_delta_ts)
            ))

        # -------- 5. Agregación segura --------
        if not fold_results:
            raise RuntimeError("No hay folds válidas (todas fueron descartadas).")

        agg = lambda k: np.mean([d[k] for d in fold_results if not np.isnan(d[k])])

        return dict(
            cluster_id=cluster_id, n_samples=len(cluster_data),
            n_folds=len(fold_results), mae_mean=agg('mae'),
            rmse_mean=agg('rmse'), r2_mean=agg('r2'),
            log_likelihood_mean=agg('ll'), pit_ks_mean=agg('pit_ks'),
            lambda_A_mean=agg('lambda_A_mean'), lambda_D_mean=agg('lambda_D_mean'),
            convergence_success=True
        )

    except Exception as e:
        print(f"❌ Error en cluster {cluster_id}: {e}")
        return dict(cluster_id=cluster_id, n_samples=len(cluster_data),
                    convergence_success=False, error=str(e))

print("✅ Per-cluster baseline function defined")

# ====================================================================
# NEW: Utilities for wide format modelling (all-cluster features)
# ====================================================================

def create_wide_feature_matrix(
    df: pl.DataFrame,
    feature_cols: List[str],
    *,
    ts_col: str = "ts_start",
    cluster_col: str = "cluster",
    fill_value: float | int = 0.0,
) -> pd.DataFrame:
    """Return a *wide* feature matrix with one row per timestamp.

    Each original feature is duplicated for every cluster so that, for
    example, the feature ``dep_last_DT`` for cluster ``3`` becomes the
    column ``dep_last_DT_3``.

    Parameters
    ----------
    df : pl.DataFrame
        Input long-format dataframe containing ``ts_col``, ``cluster_col``
        and the columns in ``feature_cols``.
    feature_cols : list[str]
        Names of the feature columns to be pivoted.
    ts_col : str, default "ts_start"
        Timestamp column to be used as the index of the resulting matrix.
    cluster_col : str, default "cluster"
        Column identifying the cluster/station group.
    fill_value : float | int, default 0.0
        Value to fill NaNs created by the pivot (i.e. timestamps where a
        given cluster does not have data).

    Returns
    -------
    pandas.DataFrame
        Wide dataframe indexed by ``ts_col`` and **sorted chronologically**.
        The number of columns equals ``len(feature_cols) * n_clusters``.
    """

    # Convert to pandas for the powerful pivot functionality
    pdf_long = df.to_pandas()

    # Keep only relevant columns to avoid holding unnecessary data in memory
    keep_cols = [ts_col, cluster_col] + feature_cols
    pdf_long = pdf_long[keep_cols].copy()

    # Ensure no duplicate index after pivot (aggregate via first() if needed)
    pdf_long = (
        pdf_long.dropna(subset=[ts_col, cluster_col])
        .sort_values(ts_col)
    )

    # Perform the pivot – resulting columns are a MultiIndex (feature, cluster)
    pdf_wide = (
        pdf_long
        .set_index([ts_col, cluster_col])[feature_cols]
        .unstack(level=cluster_col)
    )

    # Flatten the MultiIndex columns:  (feature, cluster) -> f"{feature}_{cluster}"
    pdf_wide.columns = [f"{feat}_{clust}" for feat, clust in pdf_wide.columns.to_flat_index()]

    # Order rows chronologically and fill missing values
    pdf_wide = pdf_wide.sort_index()
    pdf_wide = pdf_wide.fillna(fill_value)

    print(
        f"✅ Created wide feature matrix with shape {pdf_wide.shape} – "
        f"{pdf_wide.shape[1]} features across {len(set(df[cluster_col].to_list()))} clusters"
    )

    return pdf_wide


def build_wide_cluster_dataset(
    wide_X: pd.DataFrame,
    df_long: pl.DataFrame,
    cluster_id: int,
    target_A: str,
    target_D: str,
    ts_col: str = "ts_start",
    cluster_col: str = "cluster",
) -> pd.DataFrame:
    """Return a *pandas* dataframe ready for modelling a specific cluster.

    The returned dataframe contains all columns from *wide_X* plus the two
    target columns (arrivals & departures) for the requested *cluster_id*.
    Any timestamps without target information for that cluster are dropped
    to keep X/y aligned.
    """

    # Extract target series for the chosen cluster
    pdf_targets = (
        df_long
        .filter((pl.col(cluster_col) == cluster_id))
        .select([ts_col, target_A, target_D])
        .to_pandas()
        .set_index(ts_col)
        .sort_index()
    )

    # Align with the wide feature matrix
    common_index = wide_X.index.intersection(pdf_targets.index)
    if common_index.empty:
        raise ValueError(f"No overlapping timestamps between wide matrix and cluster {cluster_id} targets.")

    df_cluster = pd.concat([
        wide_X.loc[common_index],
        pdf_targets.loc[common_index]
    ], axis=1)

    return df_cluster

# ====================================================================
# END new utilities
# ====================================================================
