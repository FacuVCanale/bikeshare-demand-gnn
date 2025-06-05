# EcoBici-AI

Proyecto para predecir la demanda de estaciones de bicicletas en la Ciudad de Buenos Aires utilizando historiales de viajes de EcoBici.

Este repositorio provee un pipeline de procesamiento de datos y modelos de referencia para pronosticar la cantidad de arribos y partidas por estación en intervalos de tiempo discretos.

## Estructura de carpetas

```
raw/             <- CSV originales
processed/       <- Parquets con conteos por estación y features externas
   ├── dispatch_counts.parquet
   └── arrival_counts.parquet
datasets/        <- Conjuntos de entrenamiento, validación y prueba
models/          <- Modelos entrenados
reports/         <- Informes y artefactos de análisis
notebooks/       <- Exploraciones y EDA
src/             <- Código fuente de los pipelines y entrenamiento
```

## Requisitos

- Python >= 3.10
- pandas
- polars (opcional)
- scikit-learn
- pyarrow
- xgboost

Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Uso rápido

1. Colocar los CSV descargados de EcoBici en `data/raw/`.
2. Ejecutar el pipeline de procesamiento:

```bash
python src/data_pipeline.py --input_dir data/raw --output_dir data/processed
```

3. Construir los datasets para entrenamiento:

```bash
python src/build_datasets.py --processed_dir data/processed --output_dir data/datasets
```

4. Entrenar modelos base:

```bash
python src/train_baselines.py --data_dir data/datasets --models_dir models
```

## Referencias

La estructura y las etapas de este proyecto siguen la propuesta detallada en [este enlace](https://chatgpt.com/share/683f51dc-ae58-8008-b59f-2aaf9ac1d3d7).

