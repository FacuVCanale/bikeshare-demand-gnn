# EcoBici-AI: Predicción de Demanda con Graph Neural Networks

Sistema de predicción de demanda de bicicletas públicas utilizando Graph Neural Networks (GNN) para el sistema EcoBici de Buenos Aires.

## 📋 Descripción del Proyecto

Este proyecto predice la demanda de bicicletas en el sistema EcoBici utilizando:

- **Filtrado de estaciones**: Solo incluye estaciones con ≥5000 viajes totales para garantizar robustez
- **Clustering de estaciones**: Agrupa estaciones geográficamente cercanas
- **Features temporales**: Patrones horarios, diarios, semanales y estacionales  
- **Features de lag**: Histórico de demanda para capturar tendencias temporales
- **Features meteorológicos**: Temperatura, precipitación, viento, etc.
- **Graph Neural Networks**: Capturan relaciones espaciales entre clusters

### Tipos de Predicción Disponibles:
- **Arribos externos**: Viajes entre clusters (llegadas)
- **Partidas externas**: Viajes entre clusters (salidas)
- **Ambos**: Arribos + partidas simultáneamente
- **Demografia**: Predicciones por género y edad

## 🛠️ Instalación

### Requisitos
- Python 3.8+
- CUDA (opcional, para GPU)

### Dependencias Principales
```bash
pip install -r requirements.txt
```

Dependencias clave:
- `torch` + `torch-geometric`: Para GNN
- `polars`: Procesamiento eficiente de datos
- `scikit-learn`: Clustering y métricas
- `pandas`, `numpy`: Manipulación de datos
- `optuna`: Optimización de hiperparámetros

## 🚀 Workflow Completo Paso a Paso

El flujo de trabajo tiene 4 pasos principales:

### ✅ Paso 1: Generar Dataset Clusterizado
### ✅ Paso 2: Preparar Dataset para GNN  
### ✅ Paso 3: Entrenar Modelo GNN
### ✅ Paso 4: Evaluar Modelo
### 🔧 Paso 5 (Opcional): Optimización de Hiperparámetros

---

## 📊 Paso 1: Generar Dataset Clusterizado

Este paso carga los datos raw, filtra estaciones con baja actividad (<5000 viajes), crea clusters de estaciones y genera features.

### Script: `generate_cluster_dataset.py`

```bash
python scripts/generate_cluster_dataset.py [opciones]
```

### Parámetros Principales:

#### **Configuración de Datos**
- `--data_dir` (default: `data`): Directorio raíz de datos
- `--output_dir` (default: `data/clustered`): Directorio de salida

#### **Configuración de Clustering**
- `--n_clusters` (default: `93`): Número de clusters K-means. Usar `none` para features a nivel estación individual
- `--random_state` (default: `42`): Semilla para reproducibilidad

#### **Configuración Temporal**
- `--dt_minutes` (default: `30`): Intervalo temporal en minutos para agregación
- `--train_end_date` (default: `"2023-01-01"`): Fecha fin del conjunto de entrenamiento (YYYY-MM-DD)
- `--val_end_date` (default: `"2023-07-01"`): Fecha fin del conjunto de validación (YYYY-MM-DD)

#### **Opciones de Procesamiento**
- `--use_checkpoints`: Usar checkpoints para reanudar procesamiento
- `--clear_checkpoints`: Limpiar checkpoints existentes antes de empezar
- `--log_level` (default: `INFO`): Nivel de logging (DEBUG, INFO, WARNING, ERROR)

### Ejemplos de Uso:

```bash
# Configuración básica con clustering
python scripts/generate_cluster_dataset.py \
    --n_clusters 93 \
    --dt_minutes 30 \
    --train_end_date "2023-01-01" \
    --val_end_date "2023-07-01"

# Sin clustering (features por estación individual)
python scripts/generate_cluster_dataset.py \
    --n_clusters none \
    --output_dir data/station_level

# Con checkpoints para datasets grandes
python scripts/generate_cluster_dataset.py \
    --use_checkpoints \
    --log_level DEBUG
```

### Salida Generada:

**Con clustering habilitado:**
```
data/clustered/
├── train_cluster_features.parquet
├── val_cluster_features.parquet  
├── test_cluster_features.parquet
├── dataset_metadata.json
├── generation_summary.txt
└── models/kmeans_k93_model.pkl
```

**Sin clustering (--n_clusters none):**
```
data/clustered/
├── train_station_features.parquet
├── val_station_features.parquet  
├── test_station_features.parquet
├── dataset_metadata.json
└── generation_summary.txt
```

### Filtrado de Estaciones:
- Solo se incluyen estaciones con ≥5000 viajes totales (origen + destino)
- Las estaciones de baja actividad se filtran antes de cualquier procesamiento
- Las estadísticas de filtrado se incluyen en metadata y reportes

---

## 🔗 Paso 2: Preparar Dataset para GNN

Este paso convierte los datos clusterizados (o a nivel estación) al formato PyTorch Geometric, elimina features que causan data leakage y crea la estructura de grafo. El script automáticamente detecta si los datos fueron generados con clustering o a nivel estación.

### Script: `prepare_gnn_dataset.py`

```bash
python scripts/prepare_gnn_dataset.py [opciones]
```

### Parámetros Principales:

#### **Configuración de Entrada/Salida**
- `--input_dir` (default: `data/clustered`): Directorio con archivos parquet clusterizados
- `--output_dir` (default: `data/gnn_ready`): Directorio de salida para datos procesados de GNN

#### **Configuración de Target**
- `--target` (default: `arr_external_count`): Variable(s) objetivo a predecir
  - `arr_external_count`: Solo arribos externos
  - `dep_external_count`: Solo partidas externas  
  - `both_external_counts`: Arribos + partidas (recomendado)
  - `arr_external_demographics`: Demografia de arribos
  - `dep_external_demographics`: Demografia de partidas
  - `all_external_demographics`: Toda la demografia

#### **Configuración de Secuencias Temporales**
- `--sequence_length` (default: `24`): Longitud de secuencias temporales (en pasos de tiempo)

#### **Configuración de Grafo**
- `--k_neighbors` (default: `5`): Número de vecinos más cercanos para conectividad del grafo
- `--distance_threshold` (default: `5.0`): Distancia máxima (km) para conexiones entre clusters

#### **Opciones de Procesamiento**
- `--no_multiprocessing`: Deshabilitar multiprocesamiento (útil para debugging)

### Ejemplos de Uso:

```bash
# Configuración básica - solo arribos
python scripts/prepare_gnn_dataset.py \
    --target arr_external_count \
    --sequence_length 24 \
    --k_neighbors 5

# Arribos + partidas (recomendado)
python scripts/prepare_gnn_dataset.py \
    --target both_external_counts \
    --sequence_length 24 \
    --k_neighbors 5 \
    --distance_threshold 5.0

# Demografia completa
python scripts/prepare_gnn_dataset.py \
    --target all_external_demographics \
    --output_dir data/gnn_demographics

# Configuración personalizada de grafo
python scripts/prepare_gnn_dataset.py \
    --target both_external_counts \
    --k_neighbors 8 \
    --distance_threshold 3.0 \
    --sequence_length 48
```

### Salida Generada:
```
data/gnn_ready/
├── train_data.pt
├── val_data.pt  
├── test_data.pt
├── train_feature_names.json
├── val_feature_names.json
├── test_feature_names.json
├── processing_config.json
├── adjacency_matrix.npy
└── edge_index.pt
```

---

## 🧠 Paso 3: Entrenar Modelo GNN

Este paso entrena modelos Graph Neural Network usando los datos preparados.

### Script: `train_gnn.py`

```bash
python scripts/train_gnn.py [opciones]
```

### Parámetros por Categoría:

#### **Configuración del Modelo**
- `--model-type` (default: `gcn`): Tipo de modelo GNN
  - `gcn`: TemporalGCN (baseline eficiente)
  - `gat`: SpatialGAT (con mecanismo de atención)
  - `sage`: GraphSAGE (escalable)
  - `transformer`: GraphTransformer (estado del arte)
  - `hybrid`: HybridSpatioTemporalGNN (relaciones espacio-temporales)

- `--hidden-dim` (default: `64`): Dimensión de capas ocultas
- `--dropout` (default: `0.2`): Tasa de dropout
- `--num-layers` (default: `3`): Número de capas GNN (gcn, gat, sage, transformer)
- `--use-batch-norm` (default: `True`): Usar batch normalization

#### **Parámetros Específicos por Modelo**

**GAT y Transformer:**
- `--num-heads` (default: `4`): Número de cabezas de atención
- `--attention-dropout` (default: `0.1`): Dropout de atención (solo GAT)

**GraphSAGE:**
- `--aggregation` (default: `mean`): Tipo de agregación (`mean`, `max`, `sum`)

**Hybrid:**
- `--use-temporal` (default: `True`): Usar componentes temporales
- `--temporal-dim` (default: `64`): Dimensión temporal

#### **Configuración de Datos**
- `--data-path` (default: `data/processed/gnn_features.parquet`): Ruta a datos GNN procesados
- `--spatial-threshold` (default: `1000.0`): Umbral espacial para aristas del grafo (metros)
- `--temporal-window` (default: `24`): Tamaño de ventana temporal (horas)
- `--target-column` (default: `demand`): Columna objetivo para predicción
- `--val-split` (default: `0.2`): Proporción del conjunto de validación
- `--test-split` (default: `0.1`): Proporción del conjunto de test

#### **Configuración de Entrenamiento**
- `--epochs` (default: `100`): Número de epochs de entrenamiento
- `--learning-rate` (default: `0.001`): Tasa de aprendizaje
- `--weight-decay` (default: `1e-5`): Weight decay para regularización
- `--optimizer` (default: `adam`): Tipo de optimizador (`adam`, `adamw`, `sgd`)
- `--scheduler` (default: `plateau`): Scheduler de learning rate (`plateau`, `cosine`, `none`)
- `--loss-function` (default: `mse`): Función de pérdida (`mse`, `mae`, `huber`)
- `--early-stopping-patience` (default: `15`): Paciencia para early stopping

#### **Configuración del Sistema**
- `--device` (default: `auto`): Dispositivo a usar
  - `auto`: Detección automática
  - `cpu`: Forzar CPU
  - `cuda`: Usar GPU específica
  - `cuda:0`, `cuda:1`: GPU específica
  - `multi-gpu`: Usar múltiples GPUs

- `--seed` (default: `42`): Semilla aleatoria para reproducibilidad

#### **Configuración de Experimentos**
- `--experiment-name`: Nombre del experimento (default: auto-generado)
- `--output-dir` (default: `experiments`): Directorio para guardar resultados
- `--save-best-model` (default: `True`): Guardar mejor modelo durante entrenamiento
- `--save-checkpoints` (default: `True`): Guardar checkpoints periódicos
- `--checkpoint-frequency` (default: `10`): Frecuencia de guardado de checkpoints (epochs)
- `--verbose` (default: `True`): Salida detallada durante entrenamiento
- `--config-file`: Archivo JSON de configuración para cargar parámetros

### Ejemplos de Uso:

```bash
# Entrenamiento básico con GAT
python scripts/train_gnn.py \
    --model-type gat \
    --data-path data/gnn_ready \
    --epochs 100

# Transformer avanzado con hiperparámetros optimizados
python scripts/train_gnn.py \
    --model-type transformer \
    --hidden-dim 512 \
    --num-layers 4 \
    --num-heads 16 \
    --epochs 2000 \
    --early-stopping-patience 100 \
    --learning-rate 1e-4 \
    --dropout 0.15

# Entrenamiento con múltiples GPUs
python scripts/train_gnn.py \
    --model-type gat \
    --device multi-gpu \
    --hidden-dim 1024 \
    --epochs 5000

# Configuración desde archivo JSON
python scripts/train_gnn.py \
    --config-file configs/best_gat_config.json

# Experimento con nombre personalizado
python scripts/train_gnn.py \
    --model-type hybrid \
    --experiment-name "hybrid_large_temporal" \
    --use-temporal \
    --temporal-dim 128 \
    --hidden-dim 256
```

### Salida Generada:
```
experiments/gnn/experiment_name/
├── final_model.pt
├── best_model.pt
├── config.json
├── results.json
├── training_history.json
└── checkpoints/
    ├── checkpoint_epoch_10.pt
    ├── checkpoint_epoch_20.pt
    └── ...
```

---

## 🔍 Paso 4: Evaluar Modelo

Este paso evalúa modelos entrenados en el conjunto de test.

### Script: `test_model.py`

```bash
python scripts/test_model.py [opciones]
```

### Parámetros:

#### **Configuración Requerida**
- `--model_path` (REQUERIDO): Ruta al modelo guardado (.pt file)

#### **Configuración de Datos y Modelo**
- `--data_dir` (default: `data/gnn_ready`): Directorio con datos GNN preprocesados
- `--model_type`: Tipo de modelo (`gcn`, `gat`, `sage`, `transformer`, `hybrid`). Si no se especifica, intenta inferirlo del checkpoint

#### **Configuración del Sistema**
- `--device` (default: `auto`): Dispositivo a usar (`auto`, `cpu`, `cuda`, `multi-gpu`)

#### **Opciones de Salida**
- `--save_predictions`: Guardar predicciones en archivo

### Ejemplos de Uso:

```bash
# Evaluación básica
python scripts/test_model.py \
    --model_path experiments/gnn/my_experiment/final_model.pt

# Evaluación con tipo de modelo específico
python scripts/test_model.py \
    --model_path final_model.pt \
    --model_type gat \
    --data_dir data/gnn_ready

# Evaluación con múltiples GPUs y guardar predicciones
python scripts/test_model.py \
    --model_path best_model.pt \
    --device multi-gpu \
    --save_predictions

# Evaluación con datos personalizados
python scripts/test_model.py \
    --model_path transformer_model.pt \
    --data_dir data/gnn_demographics \
    --model_type transformer
```

### Salida Generada:
- Métricas en consola (MSE, MAE, RMSE, R²)
- Archivo de predicciones (si se especifica `--save_predictions`)

---

## 🔧 Paso 5 (Opcional): Optimización de Hiperparámetros

Este paso usa Optuna para optimizar automáticamente los hiperparámetros de los modelos.

### Script: `optuna_gnn_hyperopt.py`

```bash
python scripts/optuna_gnn_hyperopt.py [opciones]
```

### Parámetros por Categoría:

#### **Configuración de Datos y Modelos**
- `--data_dir` (default: `data/gnn_ready`): Directorio con datos GNN preprocesados
- `--models` (default: `['all']`): Modelos GNN a optimizar
  - Opciones: `gcn`, `gat`, `sage`, `transformer`, `hybrid`, `all`
  - Ejemplo: `--models gcn gat transformer`

#### **Parámetros de Optimización**
- `--trials` (default: `100`): Número de trials de optimización por modelo
- `--objective` (default: `val_loss`): Métrica objetivo a optimizar
  - Opciones: `val_loss`, `val_r2`, `val_rmse`, `val_mae`, `test_rmse`, `test_r2`, `test_mae`
- `--use_test_metric`: Usar métricas del conjunto de test en lugar de validación (usar con cuidado)

#### **Configuración de Entrenamiento**
- `--max_epochs` (default: `50`): Máximo número de epochs por trial
- `--patience` (default: `10`): Paciencia para early stopping
- `--device` (default: `auto`): Dispositivo de entrenamiento (`auto`, `cpu`, `cuda`, `multi-gpu`)

#### **Paralelismo y Persistencia**
- `--jobs` (default: `1`): Número de trabajos paralelos de optimización
- `--storage`: URL de base de datos para persistencia de estudios (ej: `sqlite:///optuna.db`)
- `--study_prefix` (default: `gnn_optuna`): Prefijo para nombres de estudios
- `--timeout`: Tiempo máximo de optimización en segundos

#### **Reproducibilidad y Salida**
- `--seed` (default: `42`): Semilla aleatoria para reproducibilidad
- `--save_dir` (default: `experiments/optuna`): Directorio base para guardar resultados
- `--resume`: Reanudar estudios existentes si se encuentran
- `--timestamp`: Timestamp personalizado para organizar resultados (default: auto-generado)

### Ejemplos de Uso:

```bash
# Optimización básica de todos los modelos
python scripts/optuna_gnn_hyperopt.py \
    --models all \
    --trials 100 \
    --objective val_r2

# Optimización específica con paralelismo
python scripts/optuna_gnn_hyperopt.py \
    --models gat transformer \
    --trials 200 \
    --jobs 4 \
    --objective val_loss

# Optimización con persistencia en base de datos
python scripts/optuna_gnn_hyperopt.py \
    --models gcn \
    --trials 500 \
    --storage sqlite:///optuna_studies.db \
    --resume

# Optimización rápida con métricas de test
python scripts/optuna_gnn_hyperopt.py \
    --models sage \
    --trials 50 \
    --objective test_rmse \
    --use_test_metric \
    --max_epochs 30

# Optimización intensiva con timeout
python scripts/optuna_gnn_hyperopt.py \
    --models transformer \
    --trials 1000 \
    --timeout 7200 \
    --device multi-gpu \
    --jobs 2
```

### Salida Generada:
```
experiments/optuna_YYYYMMDD_HHMMSS/
├── gcn/
│   ├── trial_001_r2_0_8534/
│   │   ├── model.pt
│   │   ├── training_history.json
│   │   ├── model_architecture.json
│   │   └── config.json
│   ├── trial_002_r2_0_8621/
│   ├── optimization_analysis.json
│   ├── trials_dataframe.csv
│   ├── best_config.json
│   └── trials_summary.json
├── gat/
├── optimization_summary.csv
└── optimization_metadata.json
```

---

## 📈 Ejemplos de Workflows Completos

### Workflow Básico (Predicción de Arribos)

```bash
# 1. Generar dataset clusterizado
python scripts/generate_cluster_dataset.py \
    --n_clusters 93 \
    --dt_minutes 30

# 2. Preparar datos para GNN
python scripts/prepare_gnn_dataset.py \
    --target arr_external_count \
    --sequence_length 24

# 3. Entrenar modelo GAT
python scripts/train_gnn.py \
    --model-type gat \
    --epochs 100 \
    --hidden-dim 128

# 4. Evaluar modelo
python scripts/test_model.py \
    --model_path experiments/gnn/*/final_model.pt
```

### Workflow Avanzado (Arribos + Partidas con Optimización)

```bash
# 1. Generar dataset con checkpoints
python scripts/generate_cluster_dataset.py \
    --n_clusters 93 \
    --use_checkpoints

# 2. Preparar datos para ambos targets
python scripts/prepare_gnn_dataset.py \
    --target both_external_counts \
    --k_neighbors 8 \
    --sequence_length 48

# 3. Optimizar hiperparámetros
python scripts/optuna_gnn_hyperopt.py \
    --models gat transformer \
    --trials 200 \
    --objective val_r2 \
    --jobs 4

# 4. Entrenar con mejores hiperparámetros
python scripts/train_gnn.py \
    --config-file experiments/optuna_*/gat/best_config.json \
    --epochs 2000

# 5. Evaluar con múltiples GPUs
python scripts/test_model.py \
    --model_path experiments/gnn/*/final_model.pt \
    --device multi-gpu \
    --save_predictions
```

### Workflow de Experimentación (Múltiples Targets)

```bash
# 1. Generar dataset base
python scripts/generate_cluster_dataset.py --use_checkpoints

# 2. Preparar múltiples variantes de datos
python scripts/prepare_gnn_dataset.py --target arr_external_count --output_dir data/gnn_arrivals
python scripts/prepare_gnn_dataset.py --target both_external_counts --output_dir data/gnn_both  
python scripts/prepare_gnn_dataset.py --target all_external_demographics --output_dir data/gnn_demo

# 3. Comparar modelos en cada variante
for target_dir in data/gnn_arrivals data/gnn_both data/gnn_demo; do
    python scripts/train_gnn.py --model-type gat --data-path $target_dir --epochs 200
    python scripts/train_gnn.py --model-type transformer --data-path $target_dir --epochs 200
done

# 4. Evaluar todos los modelos
find experiments/gnn -name "final_model.pt" -exec python scripts/test_model.py --model_path {} \;
```

---

## 📁 Estructura de Archivos Generados

```
EcoBici-AI/
├── data/
│   ├── raw/                           # Datos originales
│   ├── clustered/                     # Datos clusterizados (Paso 1)
│   │   ├── train_cluster_features.parquet
│   │   ├── val_cluster_features.parquet
│   │   ├── test_cluster_features.parquet
│   │   ├── dataset_metadata.json
│   │   └── models/kmeans_k93_model.pkl
│   └── gnn_ready/                     # Datos para GNN (Paso 2)
│       ├── train_data.pt
│       ├── val_data.pt
│       ├── test_data.pt
│       ├── *_feature_names.json
│       ├── processing_config.json
│       ├── adjacency_matrix.npy
│       └── edge_index.pt
├── experiments/                       # Resultados de entrenamiento (Paso 3)
│   ├── gnn/
│   │   └── experiment_name/
│   │       ├── final_model.pt
│   │       ├── best_model.pt
│   │       ├── config.json
│   │       ├── results.json
│   │       └── training_history.json
│   └── optuna/                        # Optimización (Paso 5)
│       └── optuna_YYYYMMDD_HHMMSS/
│           ├── gcn/, gat/, ...
│           └── optimization_summary.csv
└── scripts/                           # Scripts principales
    ├── generate_cluster_dataset.py    # Paso 1
    ├── prepare_gnn_dataset.py         # Paso 2
    ├── train_gnn.py                   # Paso 3
    ├── test_model.py                  # Paso 4
    └── optuna_gnn_hyperopt.py         # Paso 5
```

---

## 🔧 Troubleshooting

### Error: "No sequences created"
- **Causa**: Parámetros de secuencia incompatibles o datos insuficientes
- **Solución**: Reducir `sequence_length` o verificar que existen datos temporales suficientes

### Error: "Can't instantiate abstract class"  
- **Causa**: Problema con herencia de clases base
- **Solución**: Asegurar versión más reciente del código

### Error: "size mismatch" al cargar modelo
- **Causa**: Configuración del modelo no coincide con el checkpoint
- **Solución**: Especificar parámetros correctos en `test_model.py`

### Error: "No such file" en datos
- **Causa**: Archivos de datos no encontrados
- **Solución**: Verificar que existen archivos raw en `data/`:
  - `trips_with_weather.parquet`
  - `trips.parquet`
  - `users.parquet`

### Performance lento
- **GPU**: Usar `--device cuda` o `--device multi-gpu`
- **Memoria**: Reducir `--hidden-dim`, `--sequence-length` o `--num-layers`
- **Early stopping**: Usar `--early-stopping-patience` menor

---

## 📈 Tips de Optimización

### Para Mejor Performance:
- Usar GPU: `--device cuda` o `--device multi-gpu`
- Aumentar `--hidden-dim` (128, 256, 512)
- Usar `--model-type transformer` o `--model-type hybrid`
- Aumentar `--epochs` y `--early-stopping-patience`

### Para Mejor Generalización:
- Aumentar `--dropout` (0.2-0.4)
- Usar `--weight-decay` mayor (1e-4)
- Usar `--use-batch-norm`
- Cross-validation con diferentes `--seed`

### Para Datasets Grandes:
- Usar `--model-type sage` (más escalable)
- Usar `--use-checkpoints` en generación de datos
- Aumentar `--k-neighbors` gradualmente
- Usar `--no-multiprocessing` si hay problemas de memoria

### Para Optimización de Hiperparámetros:
- Empezar con `--trials 50` para pruebas rápidas
- Usar `--storage sqlite:///optuna.db` para persistencia
- Usar `--jobs` múltiples para paralelismo
- Optimizar primero `val_loss`, luego `val_r2`

---

## 🏃‍♂️ Quick Start

```bash
# Setup completo en 4 comandos
python scripts/generate_cluster_dataset.py --use_checkpoints
python scripts/prepare_gnn_dataset.py --target both_external_counts  
python scripts/train_gnn.py --model-type gat --epochs 100
python scripts/test_model.py --model_path experiments/gnn/*/final_model.pt
```

## 📝 Notas Importantes

- **Filtrado de Estaciones**: Solo estaciones con ≥5000 viajes son incluidas automáticamente
- **Reproducibilidad**: Usar mismo `--seed` en todos los pasos para resultados consistentes
- **Memoria**: Ajustar `--hidden-dim` y `--sequence-length` según RAM/VRAM disponible
- **Multi-GPU**: Automáticamente detectado cuando se especifica `--device multi-gpu`
- **Checkpoints**: Usar `--use-checkpoints` para reanudar procesamiento interrumpido

