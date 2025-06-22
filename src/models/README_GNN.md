# Graph Neural Networks for EcoBici Demand Prediction

Este módulo implementa varios tipos de redes neuronales de grafos (GNN) para predecir la demanda de bicicletas en el sistema EcoBici de Buenos Aires.

## Arquitecturas Implementadas

### 1. TemporalGCN
- **Descripción**: Red Convolucional de Grafos con características temporales
- **Uso**: Captura dependencias espaciales entre clusters usando capas GCN
- **Características**:
  - Múltiples capas GCN con batch normalization
  - Activaciones configurables (ReLU, ELU, Leaky ReLU)
  - Dropout para regularización
  - Predictor final con capas densas

### 2. SpatialGAT
- **Descripción**: Red de Atención de Grafos para relaciones espaciales
- **Uso**: Utiliza mecanismos de atención para enfocarse en clusters vecinos relevantes
- **Características**:
  - Múltiples cabezas de atención
  - Dropout de atención configurable
  - Concatenación de cabezas en capas intermedias
  - Cabeza única en la capa final

### 3. GraphSAGE
- **Descripción**: Modelo inductivo para aprendizaje en grafos
- **Uso**: Muestrea y agrega características de vecindarios, adecuado para nuevos clusters
- **Características**:
  - Agregación configurable (mean, max, lstm)
  - Aprendizaje inductivo
  - Escalable a grafos grandes

### 4. GraphTransformer
- **Descripción**: Transformer de grafos usando capas TransformerConv
- **Uso**: Aplica mecanismos de atención tipo transformer a datos estructurados en grafos
- **Características**:
  - Múltiples cabezas de atención
  - Normalización por capas
  - Activación GELU
  - Arquitectura similar a transformers

### 5. HybridSpatioTemporalGNN
- **Descripción**: Modelo híbrido que combina múltiples arquitecturas GNN
- **Uso**: Modelado comprehensivo espacio-temporal de la demanda de bicicletas
- **Características**:
  - Rama espacial con GCN + GAT
  - Procesamiento temporal con LSTM (opcional)
  - Fusión de representaciones
  - Arquitectura modular

## Uso Básico

### Creación de Modelos

```python
from src.models.gnn_models import create_gnn_model

# Crear un modelo GCN
model = create_gnn_model(
    model_type='gcn',
    num_features=100,
    num_targets=1,
    hidden_dim=128,
    num_layers=3,
    dropout=0.2
)

# Crear un modelo GAT
model = create_gnn_model(
    model_type='gat',
    num_features=100,
    num_targets=1,
    hidden_dim=128,
    num_heads=8,
    dropout=0.2
)
```

### Entrenamiento

```python
from src.training.gnn_trainer import GNNTrainer

# Inicializar trainer
trainer = GNNTrainer(
    model=model,
    device='cuda',
    experiment_name='ecobici_gnn'
)

# Configurar entrenamiento
trainer.setup_training(
    optimizer_name='adam',
    learning_rate=0.001,
    scheduler_name='plateau',
    loss_function='mse'
)

# Entrenar modelo
history = trainer.fit(
    train_data=train_data,
    val_data=val_data,
    epochs=100,
    early_stopping_patience=15
)
```

### Evaluación

```python
# Evaluar en datos de prueba
test_metrics = trainer.evaluate(test_data)
print(f"Test RMSE: {test_metrics['rmse']:.4f}")
print(f"Test R²: {test_metrics['r2']:.4f}")

# Hacer predicciones
predictions = trainer.predict(new_data)
```

## Scripts de Entrenamiento

### Preparación de Datos
```bash
# Preparar datos para GNN
python scripts/prepare_gnn_dataset.py \
    --input_dir data/clustered \
    --output_dir data/gnn_ready \
    --sequence_length 8 \
    --target arr_external_count
```

### Entrenamiento de Modelos

```bash
# Entrenar un modelo específico
python scripts/train_gnn.py \
    --model gcn \
    --epochs 100 \
    --learning_rate 0.001 \
    --save_results

# Comparar todos los modelos
python scripts/train_gnn.py \
    --model all \
    --epochs 50 \
    --save_results
```

## Estructura de Datos

### Formato de Entrada
Los modelos esperan datos en formato PyTorch Geometric `Data`:

```python
from torch_geometric.data import Data

data = Data(
    x=node_features,      # [num_nodes, num_features]
    edge_index=edges,     # [2, num_edges]
    y=targets,           # [num_nodes, num_targets]
    num_nodes=num_nodes
)
```

### Características de Nodos
- Estadísticas de lag (1, 2, 4, 8 períodos)
- Estadísticas rolling (media y desviación estándar)
- Características meteorológicas
- Características temporales (hora, día, mes)
- Características demográficas históricas

### Conectividad del Grafo
- Basada en proximidad geográfica
- K-vecinos más cercanos (default: 5)
- Umbral de distancia (default: 5km)
- Distancia haversine entre centroides de clusters

## Configuraciones Recomendadas

### Para Datasets Pequeños (< 50 clusters)
```python
config = {
    'hidden_dim': 64,
    'num_layers': 2,
    'dropout': 0.1,
    'learning_rate': 0.001
}
```

### Para Datasets Medianos (50-200 clusters)
```python
config = {
    'hidden_dim': 128,
    'num_layers': 3,
    'dropout': 0.2,
    'learning_rate': 0.001
}
```

### Para Datasets Grandes (> 200 clusters)
```python
config = {
    'hidden_dim': 256,
    'num_layers': 4,
    'dropout': 0.3,
    'learning_rate': 0.0005
}
```

## Métricas de Evaluación

Los modelos se evalúan usando:
- **MSE** (Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **R²** (Coefficient of Determination)

## Consideraciones de Rendimiento

### GPU vs CPU
- **GPU recomendada** para modelos con > 100 clusters
- **CPU suficiente** para experimentos pequeños
- Usar `device='auto'` para detección automática

### Memoria
- GAT y Transformer requieren más memoria debido a la atención
- GraphSAGE es más eficiente en memoria
- Ajustar `batch_size` si hay problemas de memoria

### Tiempo de Entrenamiento
- GCN: ~1-2 min/época
- GAT: ~2-4 min/época  
- Transformer: ~3-5 min/época
- Hybrid: ~4-6 min/época

## Troubleshooting

### Error: "CUDA out of memory"
```python
# Reducir dimensiones
hidden_dim = 64  # en lugar de 128
num_layers = 2   # en lugar de 3

# O usar CPU
device = 'cpu'
```

### Error: "No convergence"
```python
# Ajustar learning rate
learning_rate = 0.0001  # más bajo

# Aumentar paciencia
early_stopping_patience = 25
```

### Resultados pobres
```python
# Aumentar complejidad del modelo
hidden_dim = 256
num_layers = 4

# Ajustar regularización
dropout = 0.1  # reducir dropout
weight_decay = 1e-6  # reducir weight decay
```

## Extensiones Futuras

### Posibles Mejoras
1. **Atención Temporal**: Incorporar mecanismos de atención para secuencias temporales
2. **Meta-Learning**: Adaptación rápida a nuevos clusters
3. **Grafos Dinámicos**: Actualización de conectividad en tiempo real
4. **Multi-Task Learning**: Predicción simultánea de múltiples objetivos
5. **Uncertainty Quantification**: Estimación de incertidumbre en predicciones

### Nuevas Arquitecturas
- **Graph WaveNet**: Para patrones temporales complejos
- **STGCN**: Convoluciones espacio-temporales especializadas
- **GraphSAINT**: Para grafos muy grandes
- **DiffPool**: Pooling jerárquico de grafos

## Referencias

- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- [Graph Neural Networks: A Review](https://arxiv.org/abs/1901.00596)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [GraphSAGE: Inductive Representation Learning](https://arxiv.org/abs/1706.02216) 