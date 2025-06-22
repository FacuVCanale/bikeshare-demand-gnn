# EcoBici-AI: Predicción de Demanda con Graph Neural Networks

Sistema de predicción de demanda de bicicletas públicas utilizando Graph Neural Networks (GNN) para el sistema EcoBici de Buenos Aires.

## 📋 Índice

- [Descripción del Proyecto](#descripción-del-proyecto)
- [Instalación](#instalación)
- [Preparación de Datos](#preparación-de-datos)
- [Modelos Disponibles](#modelos-disponibles)
- [Scripts Principales](#scripts-principales)
- [Entrenamiento de Modelos](#entrenamiento-de-modelos)
- [Evaluación de Modelos](#evaluación-de-modelos)
- [Ejemplos de Uso](#ejemplos-de-uso)
- [Estructura del Proyecto](#estructura-del-proyecto)

## 🎯 Descripción del Proyecto

Este proyecto predice la demanda de bicicletas en el sistema EcoBici utilizando:

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

## 📊 Preparación de Datos

### Paso 1: Datos de Entrada Requeridos

Coloca en el directorio `data/` los siguientes archivos:

```
data/
├── trips_with_weather.parquet    # Viajes con datos meteorológicos
├── trips.parquet                 # Datos de viajes básicos
└── users.parquet                 # Información de usuarios
```

### Paso 2: Generar Dataset Clusterizado

```bash
python scripts/generate_cluster_dataset.py [opciones]
```

**Parámetros principales:**
- `--n_clusters`: Número de clusters (default: 93)
- `--dt_minutes`: Intervalo temporal en minutos (default: 30)
- `--train_end_date`: Fecha fin del entrenamiento (default: "2023-01-01")
- `--val_end_date`: Fecha fin de validación (default: "2023-07-01")
- `--random_state`: Semilla para reproducibilidad (default: 42)

**Ejemplo:**
```bash
python scripts/generate_cluster_dataset.py \
    --n_clusters 93 \
    --dt_minutes 30 \
    --train_end_date "2023-01-01" \
    --val_end_date "2023-07-01"
```

**Salida:**
```
data/clustered/
├── train_cluster_features.parquet
├── val_cluster_features.parquet
├── test_cluster_features.parquet
├── dataset_metadata.json
└── models/kmeans_k93_model.pkl
```

### Paso 3: Preparar Datos para GNN

```bash
python scripts/prepare_gnn_dataset.py [opciones]
```

**Parámetros de Target:**
- `arr_external_count`: Solo arribos externos
- `dep_external_count`: Solo partidas externas
- `both_external_counts`: Arribos + partidas (recomendado)
- `arr_external_demographics`: Demografia de arribos
- `dep_external_demographics`: Demografia de partidas
- `all_external_demographics`: Toda la demografia

**Otros parámetros:**
- `--input_dir`: Directorio de datos clusterizados (default: "data/clustered")
- `--output_dir`: Directorio de salida (default: "data/gnn_ready")
- `--sequence_length`: Longitud de secuencias temporales (default: 24)
- `--k_neighbors`: Vecinos para conectividad del grafo (default: 5)
- `--distance_threshold`: Distancia máxima para conexiones (default: 5.0 km)

**Ejemplo:**
```bash
python scripts/prepare_gnn_dataset.py \
    --target both_external_counts \
    --sequence_length 24 \
    --k_neighbors 5
```

**Salida:**
```
data/gnn_ready/
├── train_data.pt
├── val_data.pt
├── test_data.pt
├── train_feature_names.json
├── processing_config.json
├── adjacency_matrix.npy
└── edge_index.pt
```

## 🧠 Modelos Disponibles

### 1. **TemporalGCN** (`gcn`)
- **Descripción**: Graph Convolutional Network básico
- **Uso**: Baseline simple y eficiente
- **Parámetros específicos**:
  - `activation`: Función de activación ('relu', 'elu', 'leaky_relu', 'gelu')

### 2. **SpatialGAT** (`gat`) 
- **Descripción**: Graph Attention Network con mecanismo de atención
- **Uso**: Captura relaciones espaciales complejas
- **Parámetros específicos**:
  - `num_heads`: Número de cabezas de atención (default: 8)
  - `attention_dropout`: Dropout en atención (default: 0.1)

### 3. **GraphSAGE** (`sage`)
- **Descripción**: Sample and Aggregate para grafos grandes
- **Uso**: Escalable para grafos muy grandes
- **Parámetros específicos**:
  - `aggregation`: Tipo de agregación ('mean', 'max', 'sum')

### 4. **GraphTransformer** (`transformer`)
- **Descripción**: Transformer adaptado para grafos
- **Uso**: Estado del arte en muchas tareas de grafos
- **Parámetros específicos**:
  - `num_heads`: Cabezas de atención (default: 8)

### 5. **HybridSpatioTemporalGNN** (`hybrid`)
- **Descripción**: Combina GCN + GAT para relaciones espacio-temporales
- **Uso**: Mejor para datos con fuerte componente temporal
- **Parámetros específicos**:
  - `use_temporal`: Habilitar componente temporal
  - `temporal_dim`: Dimensión temporal (default: 64)

## 📝 Scripts Principales

### `generate_cluster_dataset.py`
Procesa datos raw y crea clusters de estaciones.

```bash
python scripts/generate_cluster_dataset.py [opciones]
```

**Parámetros completos:**
```bash
--data_dir          # Directorio de datos raw (default: "data")
--output_dir        # Directorio de salida (default: "data/clustered")
--n_clusters        # Número de clusters (default: 93)
--dt_minutes        # Intervalo temporal (default: 30)
--train_end_date    # Fin del entrenamiento (default: "2023-01-01")
--val_end_date      # Fin de validación (default: "2023-07-01")
--random_state      # Semilla (default: 42)
--log_level         # Nivel de logging (DEBUG, INFO, WARNING, ERROR)
--use_checkpoints   # Usar checkpoints para reanudar
--clear_checkpoints # Limpiar checkpoints existentes
```

### `prepare_gnn_dataset.py`
Convierte datos clusterizados al formato PyTorch Geometric.

```bash
python scripts/prepare_gnn_dataset.py [opciones]
```

**Parámetros completos:**
```bash
--input_dir           # Directorio de entrada (default: "data/clustered")
--output_dir          # Directorio de salida (default: "data/gnn_ready")
--target             # Target a predecir (ver opciones arriba)
--sequence_length    # Longitud de secuencias (default: 24)
--k_neighbors        # Vecinos en grafo (default: 5)
--distance_threshold # Distancia máx conexiones (default: 5.0)
```

### `train_gnn.py`
Entrena modelos GNN.

```bash
python scripts/train_gnn.py [opciones]
```

**Parámetros completos:**
```bash
# modelo y datos
--model              # Tipo de modelo (gcn, gat, sage, transformer, hybrid, all)
--data_dir          # Directorio de datos GNN (default: "data/gnn_ready")
--experiment_name   # Nombre del experimento

# reproducibilidad
--seed              # Semilla aleatoria (default: 42)

# entrenamiento
--epochs            # Número de epochs (default: 100)
--patience          # Early stopping patience (default: 15)
--learning_rate     # Learning rate (default: 0.001)
--weight_decay      # Weight decay (default: 1e-5)
--optimizer         # Optimizador (adam, adamw, sgd)
--scheduler         # Scheduler (plateau, cosine, none)

# arquitectura del modelo
--hidden_dim        # Dimensión oculta (default: 128)
--num_layers        # Número de capas (default: 3)
--dropout           # Dropout rate (default: 0.2)
--batch_norm        # Usar batch normalization (default: True)

# específicos por modelo
--num_heads         # Cabezas atención GAT/Transformer (default: 8)
--attention_dropout # Dropout atención GAT (default: 0.1)
--activation        # Activación GCN/SAGE (relu, elu, etc.)

# hardware
--device            # Device (cuda, cpu, auto)

# resultados
--save_results      # Guardar resultados detallados
```

### `test_model.py`
Evalúa modelos entrenados.

```bash
python scripts/test_model.py [opciones]
```

**Parámetros:**
```bash
--model_path        # Ruta al modelo (.pt) [REQUERIDO]
--data_dir          # Directorio de datos (default: "data/gnn_ready")
--model_type        # Tipo de modelo (si no se puede inferir)
--device            # Device (cuda, cpu, auto)
--save_predictions  # Guardar predicciones
--hidden_dim        # Override dimensión oculta
--num_layers        # Override número de capas
--num_heads         # Override cabezas de atención
--dropout           # Override dropout
```

## 🚀 Entrenamiento de Modelos

### Entrenamiento Básico

**1. Preparar datos:**
```bash
# generar clusters
python scripts/generate_cluster_dataset.py

# preparar para GNN con ambos targets
python scripts/prepare_gnn_dataset.py --target both_external_counts
```

**2. Entrenar modelo:**
```bash
# modelo GAT básico
python scripts/train_gnn.py --model gat --epochs 100

# modelo transformer avanzado
python scripts/train_gnn.py \
    --model transformer \
    --hidden_dim 512 \
    --num_layers 5 \
    --epochs 5000 \
    --patience 100 \
    --num_heads 8 \
    --learning_rate 3e-4 \
    --dropout 0.1
```

### Comparación de Modelos

```bash
# comparar todos los modelos
python scripts/train_gnn.py --model all --epochs 200 --save_results
```

### Entrenamiento con GPU

```bash
# forzar uso de GPU
python scripts/train_gnn.py --model gat --device cuda

# configuración optimizada para GPU
python scripts/train_gnn.py \
    --model transformer \
    --hidden_dim 1024 \
    --num_layers 6 \
    --epochs 10000 \
    --device cuda
```

## 🔍 Evaluación de Modelos

### Evaluar Modelo Entrenado

```bash
# evaluación básica
python scripts/test_model.py \
    --model_path experiments/gnn/experiment_name/final_model.pt

# con configuración específica
python scripts/test_model.py \
    --model_path final_model.pt \
    --hidden_dim 512 \
    --num_heads 8 \
    --save_predictions
```

### Métricas Disponibles
- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error  
- **RMSE**: Root Mean Squared Error
- **R²**: Coeficiente de determinación
- **Test Loss**: Loss en conjunto de prueba

## 📁 Ejemplos de Uso

### Ejemplo 1: Predicción Solo de Arribos

```bash
# 1. preparar datos
python scripts/prepare_gnn_dataset.py --target arr_external_count

# 2. entrenar
python scripts/train_gnn.py --model gat --epochs 500 --patience 50

# 3. evaluar
python scripts/test_model.py --model_path experiments/gnn/*/final_model.pt
```

### Ejemplo 2: Predicción de Arribos + Partidas

```bash
# 1. preparar datos
python scripts/prepare_gnn_dataset.py --target both_external_counts

# 2. entrenar transformer robusto
python scripts/train_gnn.py \
    --model transformer \
    --hidden_dim 512 \
    --num_layers 4 \
    --epochs 2000 \
    --patience 100 \
    --num_heads 16 \
    --learning_rate 1e-4 \
    --dropout 0.15 \
    --seed 42

# 3. evaluar
python scripts/test_model.py \
    --model_path experiments/gnn/*/final_model.pt \
    --hidden_dim 512 \
    --num_heads 16 \
    --save_predictions
```

### Ejemplo 3: Experimentación Completa

```bash
# 1. generar dataset completo
python scripts/generate_cluster_dataset.py \
    --n_clusters 93 \
    --dt_minutes 30 \
    --use_checkpoints

# 2. preparar múltiples targets
python scripts/prepare_gnn_dataset.py --target both_external_counts
python scripts/prepare_gnn_dataset.py --target all_external_demographics --output_dir data/gnn_demographics

# 3. comparar modelos en ambos targets
python scripts/train_gnn.py --model all --data_dir data/gnn_ready --save_results
python scripts/train_gnn.py --model all --data_dir data/gnn_demographics --save_results

# 4. entrenar mejores modelos con hiperparámetros optimizados
python scripts/train_gnn.py \
    --model transformer \
    --hidden_dim 768 \
    --num_layers 6 \
    --epochs 5000 \
    --patience 200 \
    --learning_rate 5e-5 \
    --experiment_name "transformer_large"
```

## 🏗️ Estructura del Proyecto

```
EcoBici-AI/
├── data/                          # Datos
│   ├── raw/                       # Datos originales
│   ├── clustered/                 # Datos clusterizados
│   └── gnn_ready/                # Datos para GNN
├── src/                           # Código fuente
│   ├── clustering/                # Clustering de estaciones
│   ├── models/                    # Modelos GNN
│   ├── training/                  # Entrenamiento
│   └── utils/                     # Utilidades
├── scripts/                       # Scripts principales
│   ├── generate_cluster_dataset.py
│   ├── prepare_gnn_dataset.py
│   ├── train_gnn.py
│   └── test_model.py
├── experiments/                   # Resultados de experimentos
│   └── gnn/
├── reports/                       # Reportes y análisis
└── models/                        # Modelos guardados
```

## 🔧 Troubleshooting

### Error: "Can't instantiate abstract class"
- **Causa**: Problema con la herencia de clases base
- **Solución**: Asegúrate de usar la versión más reciente del código

### Error: "size mismatch" al cargar modelo
- **Causa**: Configuración del modelo no coincide con el checkpoint
- **Solución**: Especifica parámetros correctos en `test_model.py`:
```bash
python scripts/test_model.py \
    --model_path model.pt \
    --hidden_dim 512 \
    --num_heads 8
```

### Error: "No such file" en datos
- **Causa**: Archivos de datos no encontrados
- **Solución**: Verifica que existen los archivos en `data/`:
  - `trips_with_weather.parquet`
  - `trips.parquet` 
  - `users.parquet`

### Performance lento
- **GPU**: Usar `--device cuda`
- **Batch size**: Reducir `--hidden_dim` o `--num_layers`
- **Early stopping**: Usar `--patience` menor

## 📈 Tips de Optimización

### Para Mejor Performance:
- Usar GPU: `--device cuda`
- Aumentar `hidden_dim` y `num_layers`
- Usar `--model transformer` o `--model hybrid`
- Aumentar `--epochs` y `--patience`

### Para Mejor Generalización:
- Aumentar `--dropout` (0.2-0.3)
- Usar `--weight_decay` mayor (1e-4)
- Usar regularización: `--batch_norm`
- Cross-validation con diferentes `--seed`

### Para Datasets Grandes:
- Usar `--model sage` (más escalable)
- Reducir `--sequence_length`
- Aumentar `--k_neighbors` gradualmente

