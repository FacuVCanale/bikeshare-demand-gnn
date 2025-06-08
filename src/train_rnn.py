"""
Entrenamiento de modelos RNN para predicción de arribos de bicicletas EcoBici.

Este módulo implementa funciones para entrenar redes neuronales recurrentes usando PyTorch
con integración completa de MLflow para tracking de experimentos.
"""

import os
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Agregar src al path
sys.path.append(str(Path(__file__).parent))
from time_series_rnn import EcoBiciTimeSeriesPredictor, BikeStationLSTM
from data_analysis import filter_data_until_date, temporal_split_data


def create_default_config() -> Dict[str, Any]:
    """
    Crear configuración por defecto para entrenamiento de RNN.
    
    Returns:
        Diccionario con configuración de experimento
    """
    return {
        'data': {
            'data_dir': 'data/raw/combined',
            'max_date': '2024-08-31',
            'train_end_date': '2023-06-30',
            'val_end_date': '2023-12-31',
            'test_end_date': '2024-08-31'
        },
        'model': {
            'delta_t': 30,
            'sequence_length': 12,
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.2,
            'device': 'auto'
        },
        'training': {
            'epochs': 50,
            'batch_size': 32,
            'learning_rate': 0.001,
            'patience': 10,
            'experiment_name': 'ecobici_rnn_default'
        }
    }


def load_config(config_path: Optional[str] = None, 
                config_name: str = 'base') -> Dict[str, Any]:
    """
    Cargar configuración desde archivo YAML.
    
    Args:
        config_path: Ruta al archivo de configuración
        config_name: Nombre de la configuración a cargar
        
    Returns:
        Diccionario con configuración
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / 'configs' / 'experiment_configs.yaml'
    
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            configs = yaml.safe_load(f)
        
        if config_name in configs:
            return configs[config_name]
        else:
            logger.warning(f"Configuración '{config_name}' no encontrada. Usando configuración por defecto.")
            return create_default_config()
    else:
        logger.warning(f"Archivo de configuración {config_path} no encontrado. Usando configuración por defecto.")
        return create_default_config()


def setup_mlflow(experiment_name: str, 
                 tracking_uri: Optional[str] = None) -> None:
    """
    Configurar MLflow para tracking de experimentos.
    
    Args:
        experiment_name: Nombre del experimento
        tracking_uri: URI del servidor de tracking (opcional)
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    
    # Crear o usar experimento existente
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
        logger.info(f"Experimento '{experiment_name}' creado con ID: {experiment_id}")
    except mlflow.exceptions.MlflowException:
        # El experimento ya existe
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        logger.info(f"Usando experimento existente '{experiment_name}' con ID: {experiment_id}")
    
    mlflow.set_experiment(experiment_name)


def log_config_to_mlflow(config: Dict[str, Any]) -> None:
    """
    Registrar configuración en MLflow.
    
    Args:
        config: Configuración del experimento
    """
    # Aplanar el diccionario anidado para MLflow
    flat_config = {}
    for section, params in config.items():
        if isinstance(params, dict):
            for key, value in params.items():
                flat_config[f"{section}.{key}"] = value
        else:
            flat_config[section] = params
    
    mlflow.log_params(flat_config)


def load_and_prepare_data(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cargar y preparar datos según configuración.
    
    Args:
        config: Configuración del experimento
        
    Returns:
        Diccionario con datos preparados
    """
    logger.info("Cargando datos...")
    
    data_dir = Path(config['data']['data_dir'])
    
    # Cargar datos combinados
    users_df = pd.read_csv(data_dir / 'users.csv')
    trips_df = pd.read_csv(data_dir / 'trips.csv')
    
    logger.info(f"Datos cargados: {len(users_df):,} usuarios, {len(trips_df):,} viajes")
    
    # Aplicar sampling si está configurado (para memory optimization)
    sample_fraction = config['data'].get('sample_fraction', 1.0)
    if sample_fraction < 1.0:
        logger.info(f"Aplicando sampling: {sample_fraction:.1%} de los datos")
        
        # Sampling estratificado por mes para mantener distribución temporal
        trips_df['fecha_origen_recorrido'] = pd.to_datetime(trips_df['fecha_origen_recorrido'], errors='coerce')
        trips_df['month'] = trips_df['fecha_origen_recorrido'].dt.to_period('M')
        
        # Sample por mes para mantener distribución temporal
        sampled_trips = []
        for month, group in trips_df.groupby('month'):
            sample_size = max(1, int(len(group) * sample_fraction))
            sampled_group = group.sample(n=sample_size, random_state=42)
            sampled_trips.append(sampled_group)
        
        trips_df = pd.concat(sampled_trips, ignore_index=True)
        trips_df = trips_df.drop('month', axis=1)
        
        # Filtrar usuarios relacionados con los viajes sampledos
        user_ids = set(trips_df['id_usuario'].dropna())
        users_df = users_df[users_df['id_usuario'].isin(user_ids)]
        
        logger.info(f"Después del sampling: {len(users_df):,} usuarios, {len(trips_df):,} viajes")
    
    # Preprocesar fechas
    users_df['fecha_alta'] = pd.to_datetime(users_df['fecha_alta'], errors='coerce')
    trips_df['fecha_origen_recorrido'] = pd.to_datetime(trips_df['fecha_origen_recorrido'], errors='coerce')
    
    # Filtrar hasta fecha máxima
    users_filtered, trips_filtered = filter_data_until_date(
        users_df, trips_df,
        max_date=config['data']['max_date'],
        verbose=True
    )
    
    # Split temporal
    data_splits = temporal_split_data(
        users_filtered, trips_filtered,
        train_end_date=config['data']['train_end_date'],
        val_end_date=config['data']['val_end_date'],
        test_end_date=config['data']['test_end_date'],
        verbose=True
    )
    
    return data_splits


def create_predictor_from_config(config: Dict[str, Any]) -> EcoBiciTimeSeriesPredictor:
    """
    Crear predictor RNN desde configuración.
    
    Args:
        config: Configuración del modelo
        
    Returns:
        Instancia del predictor
    """
    model_config = config['model']
    
    predictor = EcoBiciTimeSeriesPredictor(
        delta_t=model_config['delta_t'],
        sequence_length=model_config['sequence_length'],
        device=model_config['device']
    )
    
    return predictor


def train_and_evaluate_model(predictor: EcoBiciTimeSeriesPredictor,
                           data_splits: Dict[str, Any],
                           config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Entrenar y evaluar modelo RNN.
    
    Args:
        predictor: Instancia del predictor
        data_splits: Datos divididos en train/val/test
        config: Configuración del entrenamiento
        
    Returns:
        Diccionario con resultados del entrenamiento
    """
    training_config = config['training']
    model_config = config['model']
    
    # Preparar datos para el modelo
    logger.info("Preparando datos para RNN...")
    data = predictor.prepare_data(
        trips_train=data_splits['trips_train'],
        trips_val=data_splits['trips_val'],
        trips_test=data_splits['trips_test']
    )
    
    # Entrenar modelo
    logger.info("Iniciando entrenamiento...")
    training_results = predictor.train_model(
        data=data,
        epochs=training_config['epochs'],
        batch_size=training_config['batch_size'],
        learning_rate=training_config['learning_rate'],
        hidden_size=model_config['hidden_size'],
        num_layers=model_config['num_layers'],
        dropout=model_config['dropout'],
        patience=training_config['patience'],
        experiment_name=training_config['experiment_name']
    )
    
    # Evaluar modelo
    logger.info("Evaluando modelo...")
    eval_results = predictor.evaluate_model(data)
    
    # Combinar resultados
    results = {
        'training_results': training_results,
        'eval_results': eval_results,
        'data': data,
        'predictor': predictor
    }
    
    return results


def log_results_to_mlflow(results: Dict[str, Any]) -> None:
    """
    Registrar resultados en MLflow.
    
    Args:
        results: Resultados del entrenamiento y evaluación
    """
    training_results = results['training_results']
    eval_results = results['eval_results']
    
    # Log métricas de entrenamiento
    if 'train_losses' in training_results:
        for epoch, loss in enumerate(training_results['train_losses']):
            mlflow.log_metric("train_loss", loss, step=epoch)
    
    if 'val_losses' in training_results:
        for epoch, loss in enumerate(training_results['val_losses']):
            mlflow.log_metric("val_loss", loss, step=epoch)
    
    # Log métricas finales de evaluación
    for split in ['train', 'val', 'test']:
        if split in eval_results:
            metrics = eval_results[split]
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"{split}_{metric_name}", value)
    
    # Log modelo
    if 'predictor' in results:
        predictor = results['predictor']
        if predictor.model is not None:
            mlflow.pytorch.log_model(
                predictor.model,
                "model",
                registered_model_name=f"ecobici_rnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )


def create_visualization_plots(results: Dict[str, Any], 
                             save_dir: Optional[Path] = None) -> None:
    """
    Crear visualizaciones de los resultados.
    
    Args:
        results: Resultados del entrenamiento
        save_dir: Directorio para guardar plots (opcional)
    """
    training_results = results['training_results']
    eval_results = results['eval_results']
    
    # Plot 1: Curvas de pérdida
    if 'train_losses' in training_results and 'val_losses' in training_results:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(training_results['train_losses'], label='Training Loss', color='blue')
        plt.plot(training_results['val_losses'], label='Validation Loss', color='red')
        plt.title('Curvas de Pérdida durante Entrenamiento')
        plt.xlabel('Época')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Métricas de evaluación
        plt.subplot(1, 2, 2)
        splits = ['train', 'val', 'test']
        metrics = ['mae', 'rmse', 'r2']  
        
        x_pos = np.arange(len(splits))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            values = []
            for split in splits:
                if split in eval_results and metric in eval_results[split]:
                    values.append(eval_results[split][metric])
                else:
                    values.append(0)
            
            plt.bar(x_pos + i * width, values, width, label=metric.upper())
        
        plt.title('Métricas de Evaluación por Split')
        plt.xlabel('Split de Datos')
        plt.ylabel('Valor de Métrica')
        plt.xticks(x_pos + width, splits)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_dir:
            save_path = save_dir / 'training_results.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            mlflow.log_artifact(str(save_path))
        
        plt.show()


def run_experiment(config_name: str = 'base',
                  config_path: Optional[str] = None,
                  save_model: bool = True,
                  create_plots: bool = True) -> Dict[str, Any]:
    """
    Ejecutar experimento completo de entrenamiento RNN.
    
    Args:
        config_name: Nombre de la configuración a usar
        config_path: Ruta al archivo de configuración
        save_model: Si guardar el modelo entrenado
        create_plots: Si crear visualizaciones
        
    Returns:
        Diccionario con todos los resultados
    """
    # Cargar configuración
    config = load_config(config_path, config_name)
    logger.info(f"Configuración cargada: {config_name}")
    
    # Configurar MLflow
    experiment_name = config['training']['experiment_name']
    setup_mlflow(experiment_name)
    
    # Iniciar run de MLflow
    with mlflow.start_run(run_name=f"{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Log configuración
        log_config_to_mlflow(config)
        
        # Cargar datos
        data_splits = load_and_prepare_data(config)
        
        # Crear predictor
        predictor = create_predictor_from_config(config)
        
        # Entrenar y evaluar
        results = train_and_evaluate_model(predictor, data_splits, config)
        
        # Log resultados a MLflow
        log_results_to_mlflow(results)
        
        # Crear visualizaciones
        if create_plots:
            create_visualization_plots(results)
        
        # Guardar modelo localmente si se solicita
        if save_model:
            models_dir = Path('models')
            models_dir.mkdir(exist_ok=True)
            
            model_path = models_dir / f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
            torch.save(predictor.model.state_dict(), model_path)
            logger.info(f"Modelo guardado en: {model_path}")
        
        logger.info("✅ Experimento completado exitosamente!")
        
        return results


def main():
    """Función principal para ejecutar desde línea de comandos."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Entrenar modelo RNN para EcoBici')
    parser.add_argument('--config', type=str, default='base', 
                       help='Nombre de configuración a usar')
    parser.add_argument('--config-path', type=str, default=None,
                       help='Ruta al archivo de configuración')
    parser.add_argument('--no-save', action='store_true',
                       help='No guardar modelo localmente')
    parser.add_argument('--no-plots', action='store_true', 
                       help='No crear visualizaciones')
    
    args = parser.parse_args()
    
    try:
        results = run_experiment(
            config_name=args.config,
            config_path=args.config_path,
            save_model=not args.no_save,
            create_plots=not args.no_plots
        )
        
        print("\n🎉 Experimento completado exitosamente!")
        print(f"📊 Resultados de evaluación:")
        
        eval_results = results['eval_results']
        for split in ['train', 'val', 'test']:
            if split in eval_results:
                metrics = eval_results[split]
                print(f"   {split.upper()}:")
                for name, value in metrics.items():
                    print(f"     • {name.upper()}: {value:.4f}")
                    
    except Exception as e:
        logger.error(f"Error durante el experimento: {str(e)}")
        raise


if __name__ == "__main__":
    main() 