<div align="center">

# 🛒 Rossmann Store Sales Forecasting

### End to end retail demand forecasting — from classical time series to deep learning

[![Streamlit App](https://img.shields.io/badge/Streamlit-Time%20Series%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://rossmann-store-sales-forecasting.streamlit.app/?embed_options=light_theme)

[![Render App](https://img.shields.io/badge/Render-Deep%20Learning%20App-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://rossmann-store-sales-forecasting-1.onrender.com)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)

</div>

---

## 🧠 Overview

Accurate sales forecasting drives smarter inventory, staffing and promotion planning in retail. This project tackles the **[Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales)** problem - predicting daily sales across **1,115 stores** across Germany by building, comparing and deploying two independent forecasting pipelines:

| | |
|---|---|
| ⏳ **Classical Time Series** | SARIMAX, Prophet & TBATS — statistical models with exogenous regressors |
| 🤖 **Deep Learning** | FNN, GRU & N-HiTS sequence models, including quantile variants for uncertainty bands |

Both pipelines model sales at two granularities — aggregated (chain wide) and store level  while accounting for:

* 📅 Seasonality, trends & day of week effects
* 🎉 Promotions & public holidays
* 🏫 School holidays
* 🏪 Competition distance & store type
* 📈 Confidence / uncertainty intervals around each forecast

---

## 🌐 Live Applications

| App | Description | Link |
|---|---|---|
| ⏳ **Time Series Forecasting** | Interactive SARIMAX / Prophet dashboard with EDA, backtesting & custom 7-day forecasts | **[Launch →](https://rossmann-store-sales-forecasting.streamlit.app/?embed_options=light_theme)** |
| 🤖 **Deep Learning Forecasting** | FNN / GRU / N-HiTS predictor with aggregate & store-wise modes plus quantile uncertainty bands | **[Launch →](https://rossmann-store-sales-forecasting-1.onrender.com)** |

> ⚠️ The Render app runs on a free instance and may take up to 5 minutes to spin up on first visit.

---

## ⚙️ Models Used

### 📈 Classical Time Series
* **SARIMAX** — seasonal ARIMA with exogenous regressors (promo, holidays, etc.)
* **Prophet** — additive trend/seasonality decomposition with future forecasting
* **TBATS** — trigonometric seasonality for complex seasonal patterns

### 🤖 Deep Learning (Keras / TensorFlow)
* **FNN** — Feedforward Neural Network baseline
* **GRU** — Gated Recurrent Unit for sequential dependencies
* **N-HiTS** — Neural Hierarchical Interpolation for Time Series (custom Keras layers)
* **RNN** — used in quantile form for interval estimation

### 📊 Uncertainty Estimation
* Quantile regression variants of FNN, GRU, RNN & N-HiTS produce prediction intervals alongside point forecasts

Each model family is trained separately for **aggregate** (22 features) and **store level** (30 features) forecasting.

---

## 📁 Repository Structure

```
Rossmann-Store-Sales-Forecasting/
├── data/                              # Raw Rossmann dataset
│   ├── train.csv
│   ├── test.csv
│   └── store.csv
├── Models_training_colab_files/       # Training notebooks (Google Colab)
│   ├── Deep_Learning_models/
│   │   ├── DL_models_aggregate_sale.ipynb
│   │   └── DL_models_store_wise.ipynb
│   └── TImeseries_models/
│       ├── Timeseries_model_Aggregate_Sale.ipynb
│       └── Timeseries_model_store_wise_sale.ipynb
├── Saved_Models/                      # Trained model artifacts (.keras + scalers)
│   ├── aggregate/
│   └── store/
├── Timeseries_models_website/         # Streamlit app — SARIMAX / Prophet / TBATS
│   ├── app.py
│   └── utils.py
├── Deep_Learning_models_website/      # Streamlit app — FNN / GRU / N-HiTS
│   ├── app.py
│   ├── models/                        
│   ├── preprocessing/                 
│   └── utils/                       
├── Project_Report.pdf
├── Rossmann_Sales_Forecasting_PPT.pdf
└── README.md
```

---

## 📄 Project Documentation

* 📑 **[Presentation](./Rossmann_Sales_Forecasting_PPT.pdf)** — project walkthrough slides
* 📝 **[Report](./Project_Report.pdf)** — full methodology & results write up

---

## 🚀 Features

* 📊 Store level and aggregate (chain wide) forecasting
* 📅 Multi factor analysis (promo, holidays, competition, seasonality)
* 📉 Prediction intervals via quantile models
* 🔍 Interactive EDA, backtesting & custom N-day forecasts
* 🌐 Two independently deployed, interactive web applications

---

## 📌 Future Improvements

* 📡 Real time data integration
* 🧩 Ensembling classical & deep learning forecasts

---

<div align="center">

⭐ If you find this project useful, consider giving it a star!

</div>