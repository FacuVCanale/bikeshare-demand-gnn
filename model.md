
# 🚲 Ecobici – Forecasting Arrivals per Station (ΔT = 30 min)

This document is a **complete technical blueprint** for building, training and
serving a demand‑forecast pipeline based on a *zero‑inflated* approach plus a
probabilistic **Informer**.  
It condenses every decision justified in the previous conversation: distribution
choice, feature set, maths, and operational steps.

---

## 1  Motivation

| Issue | Consequence if ignored | Remedy |
|-------|-----------------------|--------|
| **76 % of rows are zeros** in `y_arrivals_next_DT` | MSE‑trained models collapse to ≈ 0 bicycles everywhere | Separate *occurrence* and *magnitude* with a **zero‑inflated** design |
| Sobredispersion (Var ≫ Mean) | Gaussian assumptions under‑estimate tail risk | Use **Negative Binomial (NB)** likelihood |
| Spatial correlation between stations | One‑hot `station_id` cannot generalise | Add **lat/long + station embeddings** |
| Concept‑drift 2020 → 2024 (COVID & network expansion) | Model decays in production | Calendar flags (`year`, `pandemic_flag`, etc.) + forward‑chaining CV |

---

## 2  Data & Feature Engineering

### 2·1  Raw Columns
The initial table (abridged):

```
station_id, ts_start, dep_last_DT, trip_dur_mean_last_DT, …,
arr_last_DT, y_arrivals_next_DT, weather_* …
```

### 2·2  Spatial Features  
| Feature | Construction |
|---------|--------------|
| `lat`, `lon`    | left‑join from station metadata |
| `lat_z`, `lon_z` | z‑score normalisation |
| `station_id_idx` | integer index **→ entity embedding** `ℝ^{d_id≈√N}` |

### 2·3  Temporal & Drift Flags  
`sin/cos_hour`, `sin/cos_dow`, `sin/cos_month`,  
`year`, `pandemic_flag` (2020), `expansion_flag` (≥ 2023),  
`trend_idx = seconds since first record / (60·30)`.

### 2·4  Activity & Lag Features  
`dep_last_DT`, `has_dep = 1[dep_last_DT>0]`,  
`trip_dur_mean_last_DT` *(null→0) +* `dur_is_null`,  
`dep_lag_1…6`, `arr_lag_1…6`, rolling stats
(`dep_ma_24h`, `dep_std_24h`, `dep_ratio_DT_24h`),  
`near_dep_sum_DT`, `near_dep_lag_1`.

### 2·5  Weather & Calendar Specials  
All `weather_*`, `wind_dir_sin|cos`, `precip_flag`,  
`is_holiday_ar`, `is_weekend`, `payday_flag`, `vacation_season`, `peak_commute`.

---

## 3  Zero‑Inflated Statistical Model

Let  

\[
Y_t \sim
\begin{cases}
0 &\text{w.p. } \pi_t,\\
\mathrm{NB}(\mu_t,\theta_t) &\text{w.p. } (1-\pi_t).
\end{cases}
\]

* **Gate** produces \(\pi_t = \Pr(Y_t=0)\).  
* **Informer** outputs \((\mu_t,\theta_t)\) for the NB component.

**Expected arrivals**

\[
\mathbb{E}[Y_t] = (1-\pi_t)\,\mu_t.
\]

**NB Variance**

\[
\operatorname{Var}(Y_t\mid Y_t>0)=\mu_t+\frac{\mu_t^2}{\theta_t}.
\]

---

## 4  Pipeline Steps

| Stage | Goal | Algorithm | Loss |
|-------|------|-----------|------|
| **1  Gate** | Predict \(p_t = 1-\pi_t = \Pr(Y_t>0)\) | LightGBM / shallow MLP | BCE / Focal |
| **2  Informer‑NB** | Predict \((\mu_t,\theta_t)\) **conditional on \(Y_t>0\)** | Informer (seq_len 96, label_len 48, pred_len 2) | Neg. Binomial NLL |

**Train / Val / Test**

* Train : 2020‑01‑01 → 2022‑12‑31  
* Val    : 2023‑01‑01 → 2023‑12‑31  
* Test  : 2024‑01‑01 → 2024‑08‑31

---

## 5  Pre‑processing Checklist

- [ ] Join coordinates; create `lat_z`, `lon_z`.  
- [ ] Map `station_id → station_id_idx` (0‑based).  
- [ ] Entity embedding dimension `d_id ≈ √N`.  
- [ ] Replace `trip_dur_mean_last_DT null → 0`; add `dur_is_null`.  
- [ ] Build flags `has_dep`, `has_arr`.  
- [ ] Impute other nulls (weather) via mediana diaria.  
- [ ] Generate lags & rolling windows.  
- [ ] Create calendar flags & trend.  
- [ ] Forward‑chain split.

---

## 6  Training Details

```text
Gate:
    LightGBM, num_leaves=64, class_weight="balanced", early_stopping=50.
Informer:
    d_model=256, n_heads=4, e_layers=2, d_layers=1,
    loss = NegativeBinomialLikelihood,
    batch_size = 256 (stratified 50 % positives),
    optimiser = AdamW lr=1e‑4,
    early_stop = 8 epochs on MAE(val).
```

NB‑head implementation (PyTorch):

```python
mu    = F.softplus(W_mu(h) + b_mu)          # mean
theta = F.softplus(W_th(h) + b_th) + 1e-3   # dispersion
```

---

## 7  Inference Logic

```python
p_event = gate.predict_proba(x_now)[0,1]          # probability Y>0
mu_hat, theta_hat = informer.predict(seq_hist)    # NB params

y_expected = p_event * mu_hat                     # point forecast
from scipy.stats import nbinom
nb = nbinom(n=theta_hat, p=theta_hat/(theta_hat+mu_hat))
ci80 = nb.ppf([0.1, 0.9])                         # confidence band
```

*Return* **`y_expected`** (rounded for logistics) plus alert if `p_event≥0.6`.

---

## 8  Monitoring in Production

| Metric | Trigger |
|--------|---------|
| PSI of `has_arr` vs train | > 0.20 |
| Monthly MAE per station | > 2 × baseline |
| Gate recall | < 0.75 |

Retrain both models when any threshold is crossed.

---

## 9  Deliverables

1. `prepare_features.py` – reproducible pre‑process  
2. `gate_model.pkl` – LightGBM binary  
3. `informer_nb.ckpt` – trained Informer weights  
4. `predict_arrivals.py` – serving function returning `(p, μ, θ, ŷ, ci)`  
5. `data‑card.md` & `model‑card.md` – documentation and audit trail  

---

## 10  Benefits

* Handles **structural zeros** and heavy tails explicitly.  
* Provides **probabilistic forecasts** (mean + interval).  
* Generalises across stations thanks to **entity embeddings** and
  **spatial coordinates**.  
* Robust to drift with calendar flags and forward‑chain evaluation.

---

