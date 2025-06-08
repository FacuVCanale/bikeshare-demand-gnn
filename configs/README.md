# 📋 Configuraciones de Experimentos RNN - EcoBici AI

Este directorio contiene las configuraciones para entrenar diferentes arquitecturas de redes neuronales recurrentes (RNN) para la predicción de arribos de bicicletas del sistema EcoBici.

## 🎯 Configuraciones Disponibles

### `base` - Configuración Balanceada
- **Uso**: Configuración estándar para la mayoría de casos
- **Contexto temporal**: 6 horas (12 ventanas de 30 min)
- **Complejidad**: Moderada (128 hidden units, 2 capas)
- **Tiempo de entrenamiento**: ~50 épocas
- **Recomendado para**: Primera prueba, baseline sólido

### `small_fast` - Configuración Rápida
- **Uso**: Desarrollo rápido y pruebas de concepto
- **Contexto temporal**: 3 horas (6 ventanas de 30 min)
- **Complejidad**: Baja (64 hidden units, 1 capa)
- **Tiempo de entrenamiento**: ~20 épocas
- **Recomendado para**: Debugging, iteración rápida

### `large_deep` - Configuración de Máximo Performance
- **Uso**: Búsqueda del mejor modelo posible
- **Contexto temporal**: 12 horas (24 ventanas de 30 min)
- **Complejidad**: Alta (256 hidden units, 3 capas)
- **Tiempo de entrenamiento**: ~100 épocas
- **Recomendado para**: Experimentos finales, competencias

### `short_window` - Alta Resolución Temporal
- **Uso**: Capturar patrones de corto plazo
- **Contexto temporal**: 6 horas (24 ventanas de 15 min)
- **Resolución**: 15 minutos
- **Recomendado para**: Predicciones de alta frecuencia

### `long_window` - Patrones de Largo Plazo
- **Uso**: Capturar tendencias diarias completas
- **Contexto temporal**: 24 horas (24 ventanas de 1 hora)
- **Resolución**: 1 hora
- **Recomendado para**: Planificación estratégica

### `experimental` - Configuración Avanzada
- **Uso**: Exploración de hiperparámetros
- **Parámetros**: Configuración intermedia personalizada
- **Recomendado para**: Investigación y optimización

### `production` - Configuración Optimizada
- **Uso**: Modelo final para producción
- **Características**: Optimizado para estabilidad y performance
- **Recomendado para**: Deploy en producción

### `debug` - Configuración de Debug
- **Uso**: Desarrollo y debugging del código
- **Características**: Modelo muy pequeño y rápido
- **Recomendado para**: Testing de funcionalidad

## 🚀 Cómo Usar las Configuraciones

### En Jupyter Notebook

```python
# Cambiar CONFIG_NAME en la celda correspondiente
CONFIG_NAME = 'base'  # o 'small_fast', 'large_deep', etc.
```

### Desde línea de comandos

```bash
# Entrenar con configuración específica
python src/train_rnn.py --config base

# Comparar múltiples configuraciones
python src/train_rnn.py --compare base small_fast large_deep
```

### Programáticamente

```python
from src.train_rnn import load_config, run_experiment

# Cargar configuración
config = load_config(config_name='base')

# Ejecutar experimento
results = run_experiment(config_name='base')
```

## ⚙️ Estructura de Configuración

Cada configuración tiene tres secciones principales:

### `data` - Configuración de Datos
```yaml
data:
  data_dir: "data/raw/combined"    # Directorio de datos
  max_date: "2024-08-31"          # Fecha límite de datos
  train_end_date: "2023-06-30"    # Fin del set de entrenamiento
  val_end_date: "2023-12-31"      # Fin del set de validación
  test_end_date: "2024-08-31"     # Fin del set de test
```

### `model` - Configuración del Modelo
```yaml
model:
  delta_t: 30                     # Ventana temporal (minutos)
  sequence_length: 12             # Longitud de secuencia
  hidden_size: 128               # Tamaño de capa oculta
  num_layers: 2                  # Número de capas LSTM
  dropout: 0.2                   # Tasa de dropout
  device: "auto"                 # CPU/CUDA automático
```

### `training` - Configuración de Entrenamiento
```yaml
training:
  epochs: 50                     # Número máximo de épocas
  batch_size: 32                 # Tamaño de batch
  learning_rate: 0.001           # Tasa de aprendizaje
  patience: 10                   # Paciencia para early stopping
  experiment_name: "..."         # Nombre del experimento MLflow
```

## 📊 Métricas de Evaluación

Todas las configuraciones son evaluadas usando:
- **MAE**: Error Absoluto Medio (métrica principal)
- **RMSE**: Raíz del Error Cuadrático Medio
- **R²**: Coeficiente de Determinación

## 🔄 Tracking con MLflow

Cada configuración genera automáticamente:
- Registro de hiperparámetros
- Métricas de entrenamiento y validación
- Curvas de aprendizaje
- Visualizaciones de resultados
- Modelo serializado (opcional)

Para ver los resultados:
```bash
mlflow ui
# Abrir http://localhost:5000
```

## 🛠️ Personalización

Para crear una configuración personalizada:

1. Copiar una configuración existente
2. Modificar los parámetros deseados
3. Agregar un nombre único
4. Guardar en `experiment_configs.yaml`

Ejemplo:
```yaml
mi_config_personalizada:
  data:
    # ... configuración de datos
  model:
    delta_t: 45                 # Ventana personalizada
    sequence_length: 16         # Secuencia personalizada
    # ... otros parámetros
  training:
    # ... configuración de entrenamiento
```

## ⚠️ Consideraciones

- **Memoria**: Configuraciones `large_deep` requieren más RAM
- **Tiempo**: Configuraciones con más épocas tardan más
- **GPU**: Se recomienda GPU para configuraciones complejas
- **Datos**: Verificar que `data_dir` apunte al directorio correcto

## 📚 Referencias Adicionales

- Ver `notebooks/model_training.ipynb` para ejemplos completos
- Consultar `src/train_rnn.py` para la implementación
- Revisar `src/time_series_rnn.py` para la arquitectura del modelo 