# 🌫️ Air Quality Prediction Using Machine Learning

**Organisation:** INLIGHN TECH  
**Website:** [https://www.inlighntech.com/](https://www.inlighntech.com/)  
**Project Type:** Time Series Forecasting | Environmental AI  
**Difficulty:** Medium–Advanced  

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Quick Start](#quick-start)
4. [Notebook Walkthrough](#notebook-walkthrough)
5. [Models Implemented](#models-implemented)
6. [Hyperparameter Tuning](#hyperparameter-tuning)
7. [Flask API Reference](#flask-api-reference)
8. [Dummy Data Predictions](#dummy-data-predictions)
9. [Dependencies](#dependencies)
10. [Limitations](#limitations)
11. [Conclusion](#conclusion)

---

## 📌 Project Overview

This project builds a comprehensive **PM2.5 (fine particulate matter) prediction pipeline** using historical air quality data from Indian cities. It demonstrates a complete ML workflow:

- Multi-source data preprocessing
- Time-series feature engineering (lags, rolling stats, temporal features)
- Training and comparing **8+ models** including classical ML, ARIMA, SARIMA, Prophet and LSTM
- **Hyperparameter tuning** via GridSearchCV, RandomizedSearchCV, and AIC-based selection
- REST API deployment via Flask
- Prediction on synthetic dummy data for inference validation

---

## 📂 Repository Structure

```
air_quality_project/
│
├── Air_Quality_Prediction_Enhanced.ipynb   ← Main notebook (run this first)
├── air_pollution_data.csv                  ← Dataset (UCI / CPCB)
│
├── app.py                                  ← Flask REST API
├── templates/
│   └── index.html                          ← Web dashboard (served by Flask)
│
├── model_artifacts/                        ← Created after running the notebook
│   ├── random_forest.pkl
│   ├── gradient_boosting.pkl
│   ├── rf_tuned.pkl
│   ├── gb_tuned.pkl
│   ├── feature_scaler.pkl
│   ├── feature_cols.json
│   ├── model_results.csv
│   └── prophet_model.pkl (if Prophet installed)
│
├── requirements.txt                        ← Python dependencies
└── README.md                               ← This file
```

---

## 🚀 Quick Start

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

Or in Google Colab:
```python
!pip install prophet statsmodels tensorflow scikit-learn pandas numpy matplotlib seaborn flask joblib
```

### Step 2 — Run the notebook

Open `Air_Quality_Prediction_Enhanced.ipynb` in Jupyter / Google Colab and run all cells **top to bottom**. This will:
- Preprocess and explore the data
- Engineer features
- Train all models
- Perform hyperparameter tuning
- Save model artifacts to `model_artifacts/`

### Step 3 — Start the Flask API

```bash
python app.py
```

Then open your browser at **http://localhost:5000**

---

## 📓 Notebook Walkthrough

| Section | Description |
|---------|-------------|
| 1. Setup | Library imports (pandas, sklearn, statsmodels, prophet, tensorflow) |
| 2. Load Data | Read CSV, inspect shape, columns, dtypes |
| 3. Preprocessing | Replace -200 sentinels, impute with mean, parse dates, AQI labels |
| 4. EDA | AQI distribution, top polluted cities, correlation heatmap, Delhi time series |
| 5. Feature Engineering | Lag features (1/7/30 days), rolling stats, temporal features |
| 6. Train-Test Split | Chronological 80/20 split, MinMaxScaler |
| 7. Model Training | 8 models: LR, RF, GB, ARIMA, SARIMA, Holt-Winters, Prophet, LSTM |
| 8. **Hyperparameter Tuning** | GridSearchCV (RF), RandomizedSearchCV (GB), AIC selection (ARIMA/SARIMA) |
| 9. Results Comparison | DataFrame + bar chart of MAE, RMSE, R² for all models |
| 10. Actual vs Predicted | Time series plots for every model |
| 11. Feature Importance | Default + tuned RF and GB importance charts |
| 12. 7-Day Forecast | Prophet future forecast with 95% CI |
| 13. **Dummy Data Predictions** | 3 synthetic scenarios tested on all models |
| 14. Save Artifacts | Pickle all models + scaler + metadata |
| 15. Summary | Overall benchmark table |
| 16. **Limitations** | Data, model, and deployment limitations |
| 17. **Conclusion** | Key findings, recommendations, impact |

---

## 🤖 Models Implemented

### Classical ML
| Model | Notes |
|-------|-------|
| Linear Regression | Baseline; fast, interpretable |
| Random Forest | Ensemble of 200 trees; robust to outliers |
| Gradient Boosting | Sequential boosting; captures complex non-linearity |

### Time Series
| Model | Notes |
|-------|-------|
| **ARIMA(5,1,2)** | AR(5) + I(1) + MA(2); non-seasonal univariate |
| **SARIMA(1,1,1)(1,1,1,7)** | Weekly seasonal component (s=7) |
| Holt-Winters | Triple exponential smoothing with trend + seasonal |
| Prophet | Facebook's decomposable Bayesian time-series model |

### Deep Learning
| Model | Notes |
|-------|-------|
| LSTM | 2-layer LSTM with Dropout + EarlyStopping (seq_len=30) |

---

## 🎯 Hyperparameter Tuning

### Random Forest — GridSearchCV
```python
param_grid = {
    'n_estimators'     : [100, 200],
    'max_depth'        : [None, 10, 20],
    'min_samples_split': [2, 5]
}
GridSearchCV(cv=3, scoring='neg_mean_squared_error')
```

### Gradient Boosting — RandomizedSearchCV
```python
param_dist = {
    'n_estimators'     : [100, 200, 300, 500],
    'learning_rate'    : [0.01, 0.05, 0.1, 0.2],
    'max_depth'        : [3, 4, 5, 6],
    'subsample'        : [0.7, 0.8, 1.0],
    'min_samples_split': [2, 5, 10]
}
RandomizedSearchCV(n_iter=20, cv=3)
```

### ARIMA/SARIMA — AIC-based Selection
All candidate orders are evaluated; the one with the lowest **AIC** (Akaike Information Criterion) is selected.

---

## 🌐 Flask API Reference

### Base URL: `http://localhost:5000`

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web dashboard |
| GET | `/health` | API health check, lists loaded models |
| GET | `/models` | Available models + benchmark results |
| POST | `/predict` | Predict PM2.5 from input features (JSON) |
| GET | `/predict/dummy` | Run predictions on 3 built-in scenarios |

### POST `/predict` — Example Request

```json
{
  "co": 900, "no": 8.5, "no2": 32, "o3": 25, "so2": 18,
  "pm10": 120, "nh3": 12, "aqi": 3,
  "day_of_week": 2, "month": 11, "quarter": 4, "day_of_year": 320,
  "pm2_5_lag_1": 95, "pm2_5_lag_7": 88, "pm2_5_lag_30": 75,
  "pm2_5_roll7_mean": 91, "pm2_5_roll7_std": 10.2, "pm2_5_roll30_mean": 82
}
```

### Example Response

```json
{
  "input_features": { "co": 900.0, "..." },
  "predictions": {
    "Random Forest":     { "pm2_5": 96.54, "category": "Very Poor" },
    "Gradient Boosting": { "pm2_5": 94.20, "category": "Very Poor" },
    "RF Tuned":          { "pm2_5": 97.11, "category": "Very Poor" },
    "GB Tuned":          { "pm2_5": 93.82, "category": "Very Poor" }
  },
  "recommended": {
    "model": "GB Tuned",
    "pm2_5": 93.82,
    "category": "Very Poor"
  }
}
```

### cURL Example

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"co":900,"no":8.5,"no2":32,"o3":25,"so2":18,"pm10":120,"nh3":12,"aqi":3,"day_of_week":2,"month":11,"quarter":4,"day_of_year":320,"pm2_5_lag_1":95,"pm2_5_lag_7":88,"pm2_5_lag_30":75,"pm2_5_roll7_mean":91,"pm2_5_roll7_std":10.2,"pm2_5_roll30_mean":82}'
```

---

## 🧪 Dummy Data Predictions

Three synthetic scenarios are used to validate the inference pipeline:

| Scenario | CO | PM10 | PM2.5 Lag-1 | Expected Category |
|----------|-----|------|------------|-------------------|
| Low Pollution (Clear Weather) | 420 | 35 | 28 | Good / Satisfactory |
| Moderate Pollution | 900 | 120 | 95 | Very Poor |
| High Pollution Spike (Smog) | 2800 | 380 | 310 | Severe |

Access via notebook (Section 13) or API:
```bash
curl http://localhost:5000/predict/dummy
```

---

## 📦 Dependencies

```
pandas>=1.5
numpy>=1.23
matplotlib>=3.6
seaborn>=0.12
scikit-learn>=1.2
statsmodels>=0.14
flask>=2.3
joblib>=1.3
prophet>=1.1          # optional
tensorflow>=2.12      # optional (for LSTM)
```

Install all:
```bash
pip install -r requirements.txt
```

---

## ⚠️ Limitations

1. **Single city**: Trained on Delhi data only — may not generalise to other cities
2. **No meteorological data**: Wind, temperature, humidity are absent but strongly affect pollution
3. **Mean imputation**: Replacing missing values with means is a simplistic strategy
4. **ARIMA is univariate**: Cannot use other pollutant readings as covariates
5. **Limited hyperparameter space**: Bayesian optimisation (Optuna) would yield better results
6. **No model drift detection**: Periodic retraining is needed for production
7. **No prediction intervals** for ML models (only Prophet provides uncertainty bands)

---

## 📊 Conclusion

This project demonstrates that **ensemble tree-based models (Random Forest, Gradient Boosting) consistently outperform** linear and univariate time-series models when rich lag and rolling features are available. Key takeaways:

- Lag features (especially 1-day and 7-day) are the most important predictors
- Hyperparameter tuning improved RMSE and R² for both RF and GB
- SARIMA captures weekly seasonality better than plain ARIMA
- Prophet is valuable for its uncertainty quantification and long-horizon forecasting

The full pipeline — from raw sensor data to a live REST API — is production-ready and can be extended with real-time data ingestion, multi-city modelling, and cloud deployment.

---

*Built with ❤️ by INLIGHN TECH · [support@inlighntech.com](mailto:support@inlighntech.com)*
