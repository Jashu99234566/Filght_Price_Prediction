import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import mlflow
import mlflow.xgboost
import numpy as np

# Set tracking URI
#mlflow.set_tracking_uri("http://localhost:5000")

# Load preprocessed dataset
df = pd.read_csv(r"C:\Users\OMEN\OneDrive\Desktop\Processed_Flight_Data.csv")

# Drop non-numeric columns
df = df.select_dtypes(exclude=['object'])

# Features and target
X = df.drop(columns=["Price"])
y = df["Price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Start MLflow run
with mlflow.start_run(run_name="XGBoostRegressor") as run:
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train, y_train)

    y_pred = xgb_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Log params and metrics
    mlflow.log_param("model", "XGBoostRegressor")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2_score", r2)

    # Log and register model
    mlflow.xgboost.log_model(xgb_model, artifact_path="model")

    mlflow.register_model(
        model_uri=f"runs:/{run.info.run_id}/model",
        name="FlightPriceXGBoostModel"
    )

    print(f"⚡ XGBoost Registered: RMSE = ₹{rmse:.2f}, R² = {r2:.4f}")
