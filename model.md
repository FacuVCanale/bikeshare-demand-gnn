# Proyecto EcoBici AI – Visión general

## ¿QUÉ QUEREMOS LOGRAR?

Diseñar una **tubería de predicción de demanda** para el sistema de bicicletas EcoBici que permita:

1. Anticipar si, en la próxima ventana de tiempo Δt (p.ej. 15 min), ocurrirá **al menos un arribo o partida** en un clúster de estaciones.
2. Una vez confirmado que habrá actividad, **pronosticar el conteo exacto** de arribos y partidas, junto con su distribución (género, edad, duración, etc.).

Dividimos el problema en dos modelos complementarios:

## MODELO 1 — GATE

* **Tipo:** Clasificación binaria.
* **Objetivo (y):** P(y > 0) → probabilidad de que exista al menos un evento (arribo o partida) en el próximo Δt para un clúster.
* **Entradas (X):**

  * Identidad y geometría del clúster (id, centróide, n.º de estaciones).
  * Variables temporales (hora, día de semana, estacionalidad, flags rush‑hour…).
  * Conteos actuales e históricos de actividad (interno vs. externo) y flags de «hay/ no hay» actividad.
  * Variables meteorológicas agregadas al timestamp (temperatura, lluvia, viento…).
* **Por qué existe:**

  * El 80 % de las ventanas son ceros → el Gate filtra los huecos y reduce cómputo de modelos costosos.
  * Permite calibrar un umbral y optimizar F1 / recall según la tolerancia a falsos negativos.

## MODELO 2 — INFORMER

* **Tipo:** Transformer multivariante para series temporales (arch. Informer).
* **Objetivo:** Predecir vectores continuos $arribos, partidas, género, edad, duración…$ para los próximos k pasos.
* **Entradas:** Ventanas históricas de todos los features numéricos + embedding de tiempo + embedding de clúster.
* **Funcionamiento:**

  * Se entrena sobre las series de cada clúster de forma multi‑actor (un solo modelo aprende todas).
  * Sólo se invoca cuando el Gate = 1 para evitar trabajo innecesario.

## FLUJO OPERATIVO

1. **Ingesta & feature engineering** sobre viajes, usuarios y clima.
2. **Clusterización K‑Means** (k = 93) para agrupar estaciones y reducir ruido.
3. Generación de dataset con métricas internas/externas, demografía y clima.
4. División temporal Train / Valid / Test (sin ‘leakage’).
5. **Gate** → clasifica cada tupla (clúster, Δt).
6. **Informer** → si Gate > threshold, produce pronósticos detallados.
7. Métricas, calibración, monitoreo y retro‑alimentación.

## MÉTRICAS CLAVE

* **Gate:** F1‑Score, Recall\@0.5 y AUC.
* **Informer:** MAE y RMSE por conteo, Pinball Loss para cuantiles.
* **Negocio:** nº de clústeres con sobre‑o sub‑abastecimiento.

## PRÓXIMOS PASOS

1. Consolidar lista final de features (baseline vs. avanzados).
2. Afinar hiperparámetros del Gate (Optuna, class weights, sampling).
3. Entrenar Informer con ventana óptima y validación cross‑cluster.
4. Empaquetar la solución en un pipeline reproducible (CLI + CI/CD).
5. Desplegar en batch/streaming y preparar dashboards de monitoreo.

Esta visión servirá como guía para el equipo durante las siguientes iteraciones.
