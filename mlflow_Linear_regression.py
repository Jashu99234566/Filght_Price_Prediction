import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import numpy as np

mlflow.set_tracking_uri("http://localhost:5000")

# Load data"
df = pd.read_csv(r"C:\Users\OMEN\OneDrive\Desktop\Processed_Flight_Data.csv")

# Drop non-numeric
df = df.select_dtypes(exclude=['object'])

# Split features and target
X = df.drop(columns=["Price"])
y = df["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start MLflow run
with mlflow.start_run(run_name="LinearRegression"):
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Log params & metrics
    mlflow.log_param("model", "LinearRegression")
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2_score", r2)

    # Log model
    mlflow.sklearn.log_model(model, "model")

    print(f"✅ Linear Regression logged with RMSE={rmse:.2f}, R²={r2:.4f}")