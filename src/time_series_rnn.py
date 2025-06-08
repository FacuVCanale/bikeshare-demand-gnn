"""
Modelo RNN para predicción de arribos de bicicletas EcoBici.

Este módulo implementa una red neuronal recurrente (RNN) usando PyTorch para predecir
la cantidad de arribos a cada estación basándose en las partidas históricas.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import warnings

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Agregar src al path
sys.path.append(str(Path(__file__).parent))
from data_analysis import load_combined_data, filter_data_until_date, temporal_split_data
from preprocess import preprocess_trips_data


class BikeStationLSTM(nn.Module):
    """
    Red neuronal LSTM para predicción de arribos de bicicletas.
    
    Arquitectura:
    - Input: características temporales de partidas por estación
    - LSTM layers: capturan dependencias temporales
    - Dense layers: mapeo a predicciones de arribos
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 num_stations: int = 447,
                 dropout: float = 0.2):
        super(BikeStationLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_stations = num_stations
        
        # LSTM para capturar secuencias temporales
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Layers de clasificación/regresión
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, num_stations)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        batch_size = x.size(0)
        
        # Inicializar hidden states
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Tomar el último output de la secuencia
        lstm_out = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Fully connected layers
        out = self.dropout(lstm_out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)  # (batch_size, num_stations)
        
        return out


class EcoBiciTimeSeriesPredictor:
    """
    Predictor de series temporales para arribos de bicicletas EcoBici.
    
    Funcionalidades:
    - Preprocesamiento de datos temporales
    - Entrenamiento de modelo RNN
    - Evaluación y tracking con MLflow
    - Predicciones futuras
    """
    
    def __init__(self, 
                 delta_t: int = 30,
                 sequence_length: int = 12,  # 6 horas de historia (12 * 30min)
                 device: str = 'auto'):
        """
        Inicializar predictor.
        
        Args:
            delta_t: Ventana de tiempo en minutos
            sequence_length: Longitud de secuencia temporal para RNN
            device: Dispositivo ('cpu', 'cuda', 'auto')
        """
        self.delta_t = delta_t
        self.sequence_length = sequence_length
        
        # Configurar device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        logger.info(f"Usando dispositivo: {self.device}")
        
        # Componentes del modelo
        self.model = None
        self.scaler_features = StandardScaler()
        self.scaler_targets = StandardScaler()  
        self.station_encoder = LabelEncoder()
        
        # Datos procesados
        self.station_info = None
        self.feature_names = []
        
    def _create_time_windows(self, 
                           trips_df: pd.DataFrame,
                           date_col: str = 'fecha_origen_recorrido') -> pd.DataFrame:
        """
        Crear ventanas temporales de delta_t minutos.
        
        Returns:
            DataFrame con conteos de partidas y arribos por estación y ventana temporal
        """
        logger.info(f"Creando ventanas temporales de {self.delta_t} minutos...")
        
        # memory optimization: process data in chunks if dataset is large
        df_size_mb = trips_df.memory_usage(deep=True).sum() / (1024**2)
        logger.info(f"Tamaño del dataset: {df_size_mb:.1f} MB")
        
        # if dataset is too large, sample it to prevent memory issues
        if df_size_mb > 500:  # if larger than 500MB
            logger.warning(f"Dataset muy grande ({df_size_mb:.1f} MB). Aplicando sampling para optimizar memoria...")
            sample_fraction = min(0.3, 500 / df_size_mb)  # sample to get ~500MB max
            df = trips_df.sample(frac=sample_fraction, random_state=42)
            logger.info(f"Usando {sample_fraction:.1%} de los datos ({len(df):,} registros)")
        else:
            df = trips_df.copy()
        
        # Asegurar que las fechas estén en datetime
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            
        if 'fecha_destino_recorrido' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['fecha_destino_recorrido']):
                df['fecha_destino_recorrido'] = pd.to_datetime(df['fecha_destino_recorrido'], errors='coerce')
        
        # drop rows with invalid timestamps to save memory
        initial_size = len(df)
        df = df.dropna(subset=[date_col])
        logger.info(f"Eliminados {initial_size - len(df):,} registros con fechas inválidas")
        
        # Crear ventanas temporales
        df['window_dispatch'] = df[date_col].dt.floor(f"{self.delta_t}min")
        if 'fecha_destino_recorrido' in df.columns:
            df['window_arrival'] = df['fecha_destino_recorrido'].dt.floor(f"{self.delta_t}min")
        
        # memory optimization: use only necessary columns
        dispatch_cols = ['window_dispatch', 'id_estacion_origen']
        if 'fecha_destino_recorrido' in df.columns and 'id_estacion_destino' in df.columns:
            arrival_cols = ['window_arrival', 'id_estacion_destino']
        else:
            arrival_cols = None
        
        # Conteos de partidas por estación y ventana
        dispatch_counts = (
            df[dispatch_cols].groupby(['window_dispatch', 'id_estacion_origen'])
            .size()
            .reset_index(name='dispatch_count')
            .rename(columns={'window_dispatch': 'timestamp', 'id_estacion_origen': 'station_id'})
        )
        
        # Conteos de arribos por estación y ventana
        if arrival_cols and all(col in df.columns for col in arrival_cols):
            arrival_counts = (
                df[arrival_cols].groupby(['window_arrival', 'id_estacion_destino'])
                .size()
                .reset_index(name='arrival_count')
                .rename(columns={'window_arrival': 'timestamp', 'id_estacion_destino': 'station_id'})
            )
        else:
            # Si no hay datos de destino, crear arribos vacíos
            arrival_counts = pd.DataFrame(columns=['timestamp', 'station_id', 'arrival_count'])
        
        # free memory by deleting the entire dataframe
        del df
        
        logger.info(f"Ventanas creadas - Partidas: {len(dispatch_counts):,}, Arribos: {len(arrival_counts):,}")
        
        return dispatch_counts, arrival_counts
    
    def _extract_station_features(self, trips_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extraer características de estaciones (ubicación, etc.).
        """
        logger.info("Extrayendo características de estaciones...")
        
        # Obtener información única de cada estación
        station_features = trips_df.groupby('id_estacion_origen').agg({
            'nombre_estacion_origen': 'first',
            'lat_estacion_origen': 'first', 
            'long_estacion_origen': 'first',
            'direccion_estacion_origen': 'first'
        }).reset_index()
        
        station_features.columns = ['station_id', 'station_name', 'latitude', 'longitude', 'address']
        
        # Agregar features geográficas
        station_features['lat_rad'] = np.radians(station_features['latitude'])
        station_features['lon_rad'] = np.radians(station_features['longitude'])
        
        # Distancia al centro de Buenos Aires (aproximadamente)
        ba_center_lat, ba_center_lon = -34.6037, -58.3816
        station_features['dist_to_center'] = np.sqrt(
            (station_features['latitude'] - ba_center_lat)**2 + 
            (station_features['longitude'] - ba_center_lon)**2
        )
        
        return station_features
    
    def _create_features_and_targets(self, 
                                   dispatch_counts: pd.DataFrame,
                                   arrival_counts: pd.DataFrame,
                                   station_features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crear matriz de características (X) y targets (y) para el modelo.
        
        Returns:
            X: (n_samples, sequence_length, n_features)
            y: (n_samples, n_stations) - arribos futuros por estación
        """
        logger.info("Creando características y targets...")
        
        # clean station_id data types to prevent sorting issues
        logger.info("Limpiando tipos de datos de station_id...")
        
        # Convert all station_ids to consistent type (int when possible, otherwise string)
        def clean_station_id(station_id):
            try:
                # Try to convert to int
                return int(float(str(station_id)))
            except (ValueError, TypeError):
                # If conversion fails, keep as string
                return str(station_id)
        
        # Apply cleaning to both dataframes
        dispatch_counts = dispatch_counts.copy()
        dispatch_counts['station_id'] = dispatch_counts['station_id'].apply(clean_station_id)
        
        if not arrival_counts.empty:
            arrival_counts = arrival_counts.copy()
            arrival_counts['station_id'] = arrival_counts['station_id'].apply(clean_station_id)
        
        # Obtener todas las ventanas temporales y estaciones
        all_timestamps = sorted(dispatch_counts['timestamp'].unique())
        
        # Get unique station IDs and sort them safely
        unique_stations = dispatch_counts['station_id'].unique()
        
        # Separate numeric and string station IDs for proper sorting
        numeric_stations = []
        string_stations = []
        
        for station in unique_stations:
            if isinstance(station, (int, float)) or (isinstance(station, str) and station.isdigit()):
                try:
                    numeric_stations.append(int(float(station)))
                except:
                    string_stations.append(str(station))
            else:
                string_stations.append(str(station))
        
        # Sort each group separately and combine
        all_stations = sorted(numeric_stations) + sorted(string_stations)
        
        logger.info(f"Estaciones encontradas: {len(all_stations)} ({len(numeric_stations)} numéricas, {len(string_stations)} texto)")
        
        # First, aggregate any duplicate timestamp + station_id combinations
        logger.info("Agregando conteos duplicados...")
        dispatch_aggregated = (
            dispatch_counts.groupby(['timestamp', 'station_id'])
            ['dispatch_count'].sum()
            .reset_index()
        )
        
        # Free memory
        del dispatch_counts
        
        if not arrival_counts.empty:
            arrival_aggregated = (
                arrival_counts.groupby(['timestamp', 'station_id'])
                ['arrival_count'].sum()
                .reset_index()
            )
            # Free memory
            del arrival_counts
        else:
            arrival_aggregated = arrival_counts
        
        # Create pivot tables directly (this handles missing combinations automatically)
        dispatch_pivot = dispatch_aggregated.pivot_table(
            index='timestamp',
            columns='station_id', 
            values='dispatch_count',
            fill_value=0,
            aggfunc='sum'  # In case there are still duplicates
        )
        
        # Similar para arrival_counts (si existen)
        if not arrival_aggregated.empty:
            arrival_pivot = arrival_aggregated.pivot_table(
                index='timestamp',
                columns='station_id',
                values='arrival_count', 
                fill_value=0,
                aggfunc='sum'
            )
            
            # Ensure both pivots have the same shape
            # Reindex arrival_pivot to match dispatch_pivot
            arrival_pivot = arrival_pivot.reindex(
                index=dispatch_pivot.index,
                columns=dispatch_pivot.columns,
                fill_value=0
            )
        else:
            arrival_pivot = pd.DataFrame(
                0, 
                index=dispatch_pivot.index,
                columns=dispatch_pivot.columns
            )
        
        # Agregar características temporales
        temporal_features = pd.DataFrame(index=dispatch_pivot.index)
        temporal_features['hour'] = temporal_features.index.hour
        temporal_features['day_of_week'] = temporal_features.index.dayofweek
        temporal_features['month'] = temporal_features.index.month
        temporal_features['is_weekend'] = temporal_features['day_of_week'].isin([5, 6]).astype(int)
        
        # Codificar características cíclicas
        temporal_features['hour_sin'] = np.sin(2 * np.pi * temporal_features['hour'] / 24)
        temporal_features['hour_cos'] = np.cos(2 * np.pi * temporal_features['hour'] / 24)
        temporal_features['day_sin'] = np.sin(2 * np.pi * temporal_features['day_of_week'] / 7)
        temporal_features['day_cos'] = np.cos(2 * np.pi * temporal_features['day_of_week'] / 7)
        temporal_features['month_sin'] = np.sin(2 * np.pi * temporal_features['month'] / 12)
        temporal_features['month_cos'] = np.cos(2 * np.pi * temporal_features['month'] / 12)
        
        # Crear secuencias para RNN
        X_sequences = []
        y_targets = []
        
        n_timestamps = len(all_timestamps)
        n_stations = len(all_stations)
        
        for i in range(self.sequence_length, n_timestamps):
            # Características de secuencia (últimos sequence_length pasos)
            seq_start = i - self.sequence_length
            seq_end = i
            
            # Features de dispatch para todas las estaciones en la secuencia
            dispatch_seq = dispatch_pivot.iloc[seq_start:seq_end].values  # (seq_len, n_stations)
            
            # Features temporales para la secuencia
            temporal_seq = temporal_features.iloc[seq_start:seq_end][
                ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos', 'is_weekend']
            ].values  # (seq_len, n_temporal_features)
            
            # Repetir features temporales para cada estación
            temporal_seq_expanded = np.repeat(
                temporal_seq[:, np.newaxis, :], 
                n_stations, 
                axis=1
            )  # (seq_len, n_stations, n_temporal_features)
            
            # Combinar dispatch y temporal features
            combined_features = np.concatenate([
                dispatch_seq[:, :, np.newaxis],  # (seq_len, n_stations, 1)
                temporal_seq_expanded
            ], axis=2)  # (seq_len, n_stations, 1 + n_temporal_features)
            
            # Aplanar para tener features por estación en secuencia
            # Reshape a (seq_len, n_stations * n_features)
            seq_features = combined_features.reshape(self.sequence_length, -1)
            
            X_sequences.append(seq_features)
            
            # Target: arribos en el timestamp actual (futuro)
            y_target = arrival_pivot.iloc[i].values  # (n_stations,)
            y_targets.append(y_target)
        
        X = np.array(X_sequences)  # (n_samples, seq_len, n_features)
        y = np.array(y_targets)    # (n_samples, n_stations)
        
        logger.info(f"Forma de X: {X.shape}, Forma de y: {y.shape}")
        
        return X, y, all_stations
    
    def prepare_data(self, 
                    trips_train: pd.DataFrame,
                    trips_val: pd.DataFrame, 
                    trips_test: pd.DataFrame) -> Dict[str, Any]:
        """
        Preparar datos completos para entrenamiento.
        
        Returns:
            Dict con datos procesados y divididos
        """
        logger.info("🔄 Preparando datos para entrenamiento...")
        
        # Crear ventanas temporales para cada split
        train_dispatch, train_arrival = self._create_time_windows(trips_train)
        val_dispatch, val_arrival = self._create_time_windows(trips_val)
        test_dispatch, test_arrival = self._create_time_windows(trips_test)
        
        # Extraer características de estaciones
        self.station_info = self._extract_station_features(trips_train)
        
        # Crear features y targets
        X_train, y_train, stations = self._create_features_and_targets(
            train_dispatch, train_arrival, self.station_info
        )
        X_val, y_val, _ = self._create_features_and_targets(
            val_dispatch, val_arrival, self.station_info
        )
        X_test, y_test, _ = self._create_features_and_targets(
            test_dispatch, test_arrival, self.station_info
        )
        
        # Normalizar características
        X_train_scaled = self.scaler_features.fit_transform(
            X_train.reshape(-1, X_train.shape[-1])
        ).reshape(X_train.shape)
        
        X_val_scaled = self.scaler_features.transform(
            X_val.reshape(-1, X_val.shape[-1])
        ).reshape(X_val.shape)
        
        X_test_scaled = self.scaler_features.transform(
            X_test.reshape(-1, X_test.shape[-1])
        ).reshape(X_test.shape)
        
        # Normalizar targets
        y_train_scaled = self.scaler_targets.fit_transform(y_train)
        y_val_scaled = self.scaler_targets.transform(y_val)
        y_test_scaled = self.scaler_targets.transform(y_test)
        
        return {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled, 
            'X_test': X_test_scaled,
            'y_train': y_train_scaled,
            'y_val': y_val_scaled,
            'y_test': y_test_scaled,
            'y_train_original': y_train,
            'y_val_original': y_val,
            'y_test_original': y_test,
            'stations': stations,
            'n_features': X_train_scaled.shape[-1],
            'n_stations': len(stations)
        }
    
    def train_model(self, 
                   data: Dict[str, Any],
                   epochs: int = 100,
                   batch_size: int = 32,
                   learning_rate: float = 0.001,
                   hidden_size: int = 128,
                   num_layers: int = 2,
                   dropout: float = 0.2,
                   patience: int = 10,
                   experiment_name: str = "ecobici_rnn") -> Dict[str, Any]:
        """
        Entrenar modelo RNN con MLflow tracking.
        """
        logger.info("🚀 Iniciando entrenamiento del modelo RNN...")
        
        # Configurar MLflow
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run():
            # Log parámetros
            mlflow.log_params({
                'model_type': 'LSTM',
                'delta_t_minutes': self.delta_t,
                'sequence_length': self.sequence_length,
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'dropout': dropout,
                'device': str(self.device)
            })
            
            # Crear modelo
            self.model = BikeStationLSTM(
                input_size=data['n_features'],
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_stations=data['n_stations'],
                dropout=dropout
            ).to(self.device)
            
            # Loss y optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
            
            # Convertir a tensores
            X_train = torch.FloatTensor(data['X_train']).to(self.device)
            y_train = torch.FloatTensor(data['y_train']).to(self.device)
            X_val = torch.FloatTensor(data['X_val']).to(self.device)
            y_val = torch.FloatTensor(data['y_val']).to(self.device)
            
            # Tracking de entrenamiento
            train_losses = []
            val_losses = []
            best_val_loss = float('inf')
            patience_counter = 0
            
            logger.info(f"Entrenando en {len(X_train)} muestras, validando en {len(X_val)}")
            
            for epoch in range(epochs):
                # Entrenamiento
                self.model.train()
                train_loss = 0.0
                
                # Mini-batch training
                for i in range(0, len(X_train), batch_size):
                    batch_X = X_train[i:i+batch_size]
                    batch_y = y_train[i:i+batch_size]
                    
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                train_loss /= (len(X_train) // batch_size + 1)
                
                # Validación
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val)
                    val_loss = criterion(val_outputs, y_val).item()
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                
                # Scheduler step
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Guardar mejor modelo
                    best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                
                print(f"Epoch {epoch:3d}/{epochs} | "
                          f"Train Loss: {train_loss:.4f} | "
                          f"Val Loss: {val_loss:.4f} | "
                          f"Best Val: {best_val_loss:.4f}")
                # Log métricas
                if epoch % 10 == 0:
                    
                    mlflow.log_metrics({
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'best_val_loss': best_val_loss
                    }, step=epoch)
                    
                    logger.info(f"Epoch {epoch:3d}/{epochs} | "
                              f"Train Loss: {train_loss:.4f} | "
                              f"Val Loss: {val_loss:.4f} | "
                              f"Best Val: {best_val_loss:.4f}")
                
                if patience_counter >= patience:
                    logger.info(f"Early stopping en época {epoch}")
                    break
            
            # Cargar mejor modelo
            self.model.load_state_dict(best_model_state)
            
            # Log final metrics
            mlflow.log_metrics({
                'final_train_loss': train_losses[-1],
                'final_val_loss': val_losses[-1],
                'best_val_loss': best_val_loss,
                'epochs_trained': len(train_losses)
            })
            
            # Guardar modelo en MLflow
            mlflow.pytorch.log_model(
                self.model,
                "model",
                registered_model_name="ecobici_rnn_model"
            )
            
            logger.info("✅ Entrenamiento completado!")
            
            return {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': best_val_loss,
                'epochs_trained': len(train_losses)
            }
    
    def evaluate_model(self, data: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluar modelo en conjunto de test.
        """
        logger.info("📊 Evaluando modelo...")
        
        if self.model is None:
            raise ValueError("Modelo no entrenado. Ejecutar train_model() primero.")
        
        self.model.eval()
        
        X_test = torch.FloatTensor(data['X_test']).to(self.device)
        y_test_original = data['y_test_original']
        
        with torch.no_grad():
            y_pred_scaled = self.model(X_test).cpu().numpy()
        
        # Desnormalizar predicciones
        y_pred = self.scaler_targets.inverse_transform(y_pred_scaled)
        
        # Calcular métricas
        mse = mean_squared_error(y_test_original, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_original, y_pred)
        
        # MAPE solo donde y_true > 0 para evitar división por 0
        mask = y_test_original > 0
        if mask.any():
            mape = mean_absolute_percentage_error(
                y_test_original[mask], 
                y_pred[mask]
            ) * 100
        else:
            mape = np.inf
        
        metrics = {
            'mse': mse,
            'rmse': rmse, 
            'mae': mae,
            'mape': mape
        }
        
        # Log métricas en MLflow
        mlflow.log_metrics({f'test_{k}': v for k, v in metrics.items()})
        
        logger.info("Métricas de evaluación:")
        for metric, value in metrics.items():
            logger.info(f"  {metric.upper()}: {value:.4f}")
            
        return metrics, y_pred
    
    def plot_predictions(self, 
                        y_true: np.ndarray, 
                        y_pred: np.ndarray,
                        stations: List[int],
                        save_path: Optional[Path] = None) -> None:
        """
        Visualizar predicciones vs valores reales.
        """
        # Seleccionar algunas estaciones para visualizar
        top_stations_idx = np.argsort(y_true.sum(axis=0))[-5:]  # Top 5 estaciones más activas
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('🚲 Predicciones vs Valores Reales - Top 5 Estaciones', fontsize=16)
        
        for i, station_idx in enumerate(top_stations_idx):
            row, col = i // 3, i % 3
            if row < 2 and col < 3:
                ax = axes[row, col]
                
                station_id = stations[station_idx]
                y_true_station = y_true[:, station_idx]
                y_pred_station = y_pred[:, station_idx]
                
                # Scatter plot
                ax.scatter(y_true_station, y_pred_station, alpha=0.6, s=20)
                
                # Línea perfecta
                max_val = max(y_true_station.max(), y_pred_station.max())
                ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.8)
                
                ax.set_xlabel('Valores Reales')
                ax.set_ylabel('Predicciones')
                ax.set_title(f'Estación {station_id}')
                ax.grid(True, alpha=0.3)
        
        # Remover subplot vacío
        if len(top_stations_idx) == 5:
            fig.delaxes(axes[1, 2])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            mlflow.log_artifact(str(save_path))
        
        plt.show()
    
    def predict_future(self, 
                      last_sequence: np.ndarray,
                      steps_ahead: int = 1) -> np.ndarray:
        """
        Hacer predicciones futuras basándose en la última secuencia observada.
        
        Args:
            last_sequence: Última secuencia observada (sequence_length, n_features)
            steps_ahead: Número de pasos futuros a predecir
            
        Returns:
            Predicciones para steps_ahead ventanas temporales futuras
        """
        if self.model is None:
            raise ValueError("Modelo no entrenado.")
        
        self.model.eval()
        
        # Preparar input
        sequence = torch.FloatTensor(last_sequence).unsqueeze(0).to(self.device)  # (1, seq_len, features)
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(steps_ahead):
                # Predecir siguiente paso
                pred_scaled = self.model(sequence)
                pred = self.scaler_targets.inverse_transform(pred_scaled.cpu().numpy())
                predictions.append(pred[0])  # Remover dimensión batch
                
                # Para predicciones multi-step, necesitaríamos actualizar la secuencia
                # Por simplicidad, mantenemos la secuencia original
                
        return np.array(predictions) 


def main():
    """Función principal para entrenar el modelo."""
    logger.info("🚀 Iniciando entrenamiento de EcoBici RNN...")
    
    # Rutas de datos
    data_dir = Path("../data/raw/combined")
    
    # Cargar datos
    logger.info("📊 Cargando datos...")
    users_df, trips_df = load_combined_data(data_dir)
    
    # Filtrar hasta agosto 2024
    users_filtered, trips_filtered = filter_data_until_date(
        users_df, trips_df, 
        max_date="2024-08-31",
        verbose=True
    )
    
    # Split temporal
    data_splits = temporal_split_data(
        users_filtered, trips_filtered,
        train_end_date="2023-06-30",
        val_end_date="2023-12-31", 
        test_end_date="2024-08-31",
        verbose=True
    )
    
    # Inicializar predictor
    predictor = EcoBiciTimeSeriesPredictor(
        delta_t=30,  # 30 minutos
        sequence_length=12,  # 6 horas de historia
        device='auto'
    )
    
    # Preparar datos
    processed_data = predictor.prepare_data(
        data_splits['trips_train'],
        data_splits['trips_val'],
        data_splits['trips_test']
    )
    
    # Entrenar modelo
    training_results = predictor.train_model(
        processed_data,
        epochs=50,
        batch_size=32,
        learning_rate=0.001,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        patience=10
    )
    
    # Evaluar modelo
    metrics, predictions = predictor.evaluate_model(processed_data)
    
    # Visualizar resultados
    predictor.plot_predictions(
        processed_data['y_test_original'],
        predictions, 
        processed_data['stations'],
        save_path=Path("../reports/rnn_predictions.png")
    )
    
    logger.info("✅ Proceso completado exitosamente!")


if __name__ == "__main__":
    main()