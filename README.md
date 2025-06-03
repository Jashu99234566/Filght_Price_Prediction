# ✈️ Unified ML App: Flight Price & Customer Satisfaction Prediction

## 📟 Overview

This project combines two machine learning pipelines into a single Streamlit application:

1. **Flight Price Prediction** using regression models (Linear Regression, Random Forest, XGBoost)
2. **Passenger Satisfaction Prediction** using classification models

It offers interactive **EDA**, **model-based predictions**, and **MLflow tracking** for reproducibility and experiment comparison.

---

## 🧠 Key Features

* 📈 **Flight Price Prediction** with 3 regression models (Linear, Random Forest, XGBoost)
* 😀 **Customer Satisfaction Classifier** using Random Forest
* 📊 **EDA modules** for both flight and passenger datasets
* 🧪 **MLflow integration** for experiment logging and model tracking
* 🧠 **One Unified Streamlit App** with sidebar navigation
* 📁 Modular structure: separate files for models, EDA, Streamlit, and training

---

## ⚙️ Tech Stack

| Area                | Libraries & Tools           |
| ------------------- | --------------------------- |
| **Frontend**        | Streamlit                   |
| **EDA**             | Seaborn, Matplotlib, pandas |
| **Modeling**        | Scikit-learn, XGBoost       |
| **Experimentation** | MLflow                      |
| **Persistence**     | Pickle, Joblib              |

---

## 📂 Project Structure

```
├── app_unified.py                 # Unified Streamlit app (main entry)
├── eda.py                         # EDA functions for flight data
├── Linear_regression.py          # Basic Linear Regression model
├── mlflow_Linear_regression.py   # MLflow-logged Linear Regression
├── Randomforest.py               # MLflow-logged Random Forest model
├── train_xgboost.py              # MLflow-logged XGBoost model
├── streamlit_app.py              # Standalone version for flight prediction
├── flightdata.ipynb              # EDA & model dev (Jupyter notebook)
├── best_model.pkl                # Pickled final model for Streamlit app
```

---

## 🚀 How to Run

1. Ensure all required files (CSV data, models) are in the expected paths.
2. Launch MLflow server if needed:

```bash
mlflow ui
```

3. Install dependencies:

```bash
pip install streamlit pandas scikit-learn xgboost mlflow matplotlib seaborn
```

4. Start the app:

```bash
streamlit run app_unified.py
```

---

## 🧪 Tracked Models (MLflow)

| Model             | RMSE (±) | R² Score | Logged |
| ----------------- | -------- | -------- | ------ |
| Linear Regression | \~4100   | \~0.62   | ✅      |
| Random Forest     | \~2900   | \~0.82   | ✅      |
| XGBoost           | \~2750   | \~0.84   | ✅      |

---

## 📌 Sample Inputs

* ✈️ Flight Duration, Departure Hour, Arrival Hour, Total Stops
* 😀 Passenger Age, In-flight Wifi, Seat Comfort, On-board Service, Flight Distance

---

## 🎉 Final Deliverable

An integrated ML dashboard that brings together regression and classification workflows in a unified Streamlit app, built for real-world airline data applications.
