# EcoBici-AI

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)
![Linting: Ruff](https://img.shields.io/badge/linting-ruff-red.svg)

Proyecto para predecir la demanda de estaciones de bicicletas en la Ciudad de Buenos Aires utilizando historiales de viajes de EcoBici.

Este repositorio provee un pipeline de procesamiento de datos y modelos de machine learning avanzados para pronosticar la cantidad de arribos y partidas por estación en intervalos de tiempo discretos.

## 🚀 Características

- ⚡ **Pipeline de datos moderno** con Pandas y Polars
- 🤖 **Múltiples algoritmos ML**: XGBoost, LightGBM, CatBoost, PyTorch, TensorFlow
- 🔍 **Optimización de hiperparámetros** con Optuna
- 📊 **Visualizaciones interactivas** con Plotly y Seaborn
- 🧪 **Testing automatizado** con pytest y coverage
- 🎯 **Linting y formateo** con Ruff, Black, e isort
- 📝 **Type checking** con MyPy
- 🛠️ **Comandos CLI** integrados

## 📁 Estructura del Proyecto

```
EcoBici-AI/
├── data/
│   ├── raw/                 <- CSV originales de EcoBici
│   ├── processed/           <- Parquets con conteos y features
│   │   ├── dispatch_counts.parquet
│   │   └── arrival_counts.parquet
│   └── datasets/            <- Conjuntos train/val/test
├── models/                  <- Modelos entrenados (.json, .pkl)
├── reports/                 <- Informes y análisis
├── notebooks/               <- Jupyter notebooks para EDA
├── src/                     <- Código fuente
│   ├── data_pipeline.py     <- Pipeline de procesamiento
│   ├── build_datasets.py    <- Construcción de datasets
│   ├── train_baselines.py   <- Modelos baseline
│   └── train_xgboost.py     <- Modelo XGBoost avanzado
├── tests/                   <- Tests unitarios e integración
├── pyproject.toml          <- Configuración del proyecto
└── README.md               <- Este archivo
```

## 🛠️ Instalación y Configuración

### Requisitos Previos

- Python >= 3.10
- Git

### Instalación para Desarrollo

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/EcoBici-AI.git
cd EcoBici-AI

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalación completa para desarrollo
pip install -e .[dev]

# Configurar pre-commit hooks (opcional pero recomendado)
pre-commit install
```

### Instalación Solo para Uso

```bash
pip install -e .
```

### Opciones de Instalación Específicas

```bash
# Solo testing
pip install -e .[test]

# Solo linting
pip install -e .[lint]

# Solo documentación
pip install -e .[docs]

# Solo notebooks
pip install -e .[notebook]
```

## 🚀 Uso Rápido

### 1. Preparar los Datos

Coloca los CSV descargados de EcoBici en `data/raw/`.

### 2. Ejecutar Pipeline Completo

```bash
# Procesar datos crudos
ecobici-pipeline --input_dir data/raw --output_dir data/processed

# Construir datasets de entrenamiento
ecobici-build-datasets --processed_dir data/processed --output_dir data/datasets

# Entrenar modelos baseline
ecobici-train-baselines --data_dir data/datasets --models_dir models

# Entrenar modelo XGBoost optimizado
ecobici-train-xgboost --data_dir data/datasets --models_dir models
```

### 3. Uso Alternativo (Comandos Python Directos)

```bash
# Pipeline de procesamiento
python -m src.data_pipeline --input_dir data/raw --output_dir data/processed

# Construcción de datasets
python -m src.build_datasets --processed_dir data/processed --output_dir data/datasets

# Entrenamiento de modelos
python -m src.train_baselines --data_dir data/datasets
python -m src.train_xgboost --data_dir data/datasets --models_dir models
```

## 🧪 Desarrollo y Testing

### Linting y Formateo

```bash
# Formatear código automáticamente
black .
isort .

# Linting con Ruff (muy rápido)
ruff check . --fix

# Type checking
mypy .

# Ejecutar todo junto
ruff check . --fix && black . && isort . && mypy .
```

### Testing

```bash
# Ejecutar todos los tests
pytest

# Tests con coverage
pytest --cov=src --cov-report=html

# Tests paralelos (más rápido)
pytest -n auto

# Solo tests rápidos
pytest -m "not slow"
```

### Pre-commit Hooks

```bash
# Instalar hooks (ejecuta linting automáticamente en cada commit)
pre-commit install

# Ejecutar manualmente en todos los archivos
pre-commit run --all-files
```

## 📊 Modelos Disponibles

### Modelos Baseline
- **Persistencia**: Usa el último valor conocido
- **Media por hora**: Promedio histórico por hora del día y día de la semana

### Modelos Avanzados
- **XGBoost**: Gradient boosting optimizado
- **LightGBM**: Gradient boosting eficiente
- **CatBoost**: Manejo automático de features categóricas
- **PyTorch**: Redes neuronales deep learning
- **TensorFlow/Keras**: Modelos de deep learning

### Optimización
- **Optuna**: Optimización automática de hiperparámetros
- **Cross-validation**: Validación cruzada temporal
- **Feature engineering**: Construcción automática de características

## 📈 Análisis y Visualizaciones

### Notebooks Jupyter

```bash
# Iniciar Jupyter Lab
jupyter lab

# Iniciar Jupyter Notebook clásico
jupyter notebook
```

### Visualizaciones Disponibles

- **Plotly**: Gráficos interactivos de demanda por estación
- **Seaborn**: Análisis estadísticos y correlaciones
- **Matplotlib**: Visualizaciones estáticas de alta calidad

## 🔧 Configuración Avanzada

### Variables de Entorno

```bash
# Configurar para producción
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Configurar logging
export LOG_LEVEL=INFO
```

### Configuración de Herramientas

Todas las herramientas están configuradas en `pyproject.toml`:

- **Ruff**: Linting ultra-rápido
- **Black**: Formateo de código
- **isort**: Ordenamiento de imports
- **MyPy**: Type checking
- **pytest**: Testing framework
- **Coverage**: Análisis de cobertura de tests

## 📝 Contribuir

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Instala dependencias de desarrollo (`pip install -e .[dev]`)
4. Ejecuta tests y linting
5. Commit tus cambios (`git commit -am 'Agrega nueva funcionalidad'`)
6. Push a la rama (`git push origin feature/nueva-funcionalidad`)
7. Abre un Pull Request

### Estándares de Código

- **Type hints**: Usar anotaciones de tipo en todas las funciones
- **Docstrings**: Documentar funciones con formato Google
- **Tests**: Escribir tests para nueva funcionalidad
- **Linting**: El código debe pasar todas las verificaciones de Ruff y MyPy

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

## 🔗 Referencias

- [Documentación de EcoBici](https://www.buenosaires.gob.ar/ecobici)
- [Datos Abiertos GCBA](https://data.buenosaires.gob.ar/)
- [Propuesta Original del Proyecto](https://chatgpt.com/share/683f51dc-ae58-8008-b59f-2aaf9ac1d3d7)

## 👨‍💻 Autor

**Juan Francisco Lebrero**
- Email: lebrerojuanfrancisco@gmail.com
- GitHub: [@tu-usuario](https://github.com/tu-usuario)

---

⭐ ¡Si este proyecto te resulta útil, no olvides darle una estrella!