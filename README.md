# 📈 End-to-End Sales Forecasting & Demand Intelligence System

## 📌 Project Overview

This project is an end-to-end Sales Forecasting and Demand Intelligence System built using Python and Machine Learning.

The system performs:

- 📊 Exploratory Data Analysis (EDA)
- 📈 Time Series Forecasting
- 🤖 Machine Learning Forecasting
- 🚨 Anomaly Detection
- 📦 Product Demand Segmentation
- 🌐 Interactive Streamlit Dashboard

The project was developed using the **Superstore Sales Dataset** and includes multiple forecasting techniques such as **SARIMA**, **Facebook Prophet**, and **XGBoost**.

---

# 📂 Project Structure

```text
SalesForecasting_Sparsh/

│── analysis.ipynb
│── app.py
│── requirements.txt
│── summary.docx
│── train.csv
│── README.md
│
└── charts/
    ├── category_region_forecast.png
    ├── elbow_method.png
    ├── isolation_forest.png
    ├── monthly_sales.png
    ├── monthly_sales_trend.png
    ├── product_clusters.png
    ├── prophet_components.png
    ├── prophet_forecast.png
    ├── region_sales.png
    ├── sarima_forecast.png
    ├── time_series_decomposition.png
    ├── xgboost_forecast.png
    └── zscore_anomalies.png
```

---

# 🛠️ Prerequisites

Before running this project, make sure you have installed:

- Python 3.10 or later
- Git (optional)
- Anaconda or Miniconda (recommended)

---

# 🚀 Create a Conda Environment

Open **Anaconda Prompt** or your terminal and run:

```bash
conda create -n salesforecast python=3.11
```

Activate the environment:

### Windows

```bash
conda activate salesforecast
```

### Linux / macOS

```bash
conda activate salesforecast
```

---

# 📦 Install Project Requirements

Install all required Python libraries:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Streamlit Dashboard

Launch the application using:

```bash
streamlit run app.py
```

After a few seconds, Streamlit will automatically open in your default web browser.

If it does not open automatically, copy and paste the displayed Local URL into your browser.

Example:

```text
Local URL: http://localhost:8501
```

---

# 📊 Dashboard Features

The dashboard contains four interactive sections:

### 🏠 Sales Overview

- KPI Cards
- Monthly Sales Trend
- Sales by Category
- Sales by Region
- Top Selling Products

---

### 📈 Forecast Explorer

- Model Performance Comparison
- XGBoost Forecast
- Category & Region Forecast

---

### 🚨 Anomaly Report

- Isolation Forest Results
- Z-Score Results
- Business Interpretation

---

### 📦 Product Demand Segmentation

- K-Means Clustering
- PCA Visualization
- Recommended Stocking Strategy

---

# 🤖 Machine Learning Models Used

## Forecasting

- SARIMA
- Facebook Prophet
- XGBoost Regressor

## Anomaly Detection

- Isolation Forest
- Z-Score Detection

## Clustering

- K-Means
- PCA

---

# 📚 Python Libraries

- Streamlit
- Pandas
- NumPy
- Matplotlib
- Plotly
- Scikit-learn
- Statsmodels
- Prophet
- XGBoost
- SciPy
- Pillow

---

# 📈 Best Forecasting Model

The three forecasting models were evaluated using MAE, RMSE, and MAPE.

| Model | MAE | RMSE | MAPE |
|------|------:|------:|------:|
| SARIMA | 18031.40 | 19009.18 | 0.19 |
| Prophet | 20250.79 | 22318.41 | 0.22 |
| **XGBoost** | **13915.32** | **18893.85** | **0.13** |

**Selected Model:** XGBoost

---

# 📄 Files Included

- ✅ analysis.ipynb
- ✅ app.py
- ✅ requirements.txt
- ✅ summary.docx
- ✅ train.csv
- ✅ charts/
- ✅ README.md

---

# 👨‍💻 Author

**Mohammad Abdullah Alam**

---

# 📜 License

This project was developed for educational and internship purposes.