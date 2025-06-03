import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load preprocessed dataset
df = pd.read_csv(r"C:\Users\OMEN\OneDrive\Desktop\Processed_Flight_Data.csv")  # If saved, else use df_final

# Print column names to check presence of string columns
print("Column types:\n", df.dtypes)

# Drop string columns if they exist
for col in df.columns:
    if df[col].dtype == 'object':
        print(f"Dropping non-numeric column: {col}")
        df.drop(columns=[col], inplace=True)

# Define features and target
X = df.drop(columns=["Price"])
y = df["Price"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict
y_pred = lr_model.predict(X_test)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Output
print(f"✅ Linear Regression RMSE: ₹{rmse:.2f}")
print(f"✅ Linear Regression R² Score: {r2:.4f}")
