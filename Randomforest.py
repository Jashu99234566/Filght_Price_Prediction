import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import numpy as np

# Set tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Load preprocessed dataset
df = pd.read_csv(r"C:\Users\OMEN\OneDrive\Desktop\Processed_Flight_Data.csv")

# Drop non-numeric columns
df = df.select_dtypes(exclude=['object'])

# Features and target
X = df.drop(columns=["Price"])
y = df["Price"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Start MLflow run
with mlflow.start_run(run_name="RandomForestRegressor"):
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Log parameters and metrics
    mlflow.log_param("model", "RandomForestRegressor")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2_score", r2)

    # Log model
    mlflow.sklearn.log_model(rf_model, "model")

    print(f"🌲 Logged RF Model: RMSE = ₹{rmse:.2f}, R² = {r2:.4f}")
