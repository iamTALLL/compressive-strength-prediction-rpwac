import joblib
import pandas as pd
from sklearn.metrics import r2_score
import sys

# Load model và scaler
try:
    scaler = joblib.load('scaler.pkl')
    model = joblib.load('xgb_woa_best_model.pkl')
except FileNotFoundError as e:
    print(f"Error: Model or scaler file not found: {e}")
    sys.exit(1)

# Đọc dữ liệu
df = pd.read_csv("Data1.csv")

# Giả sử cột cuối cùng là output thật (ground truth)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Scale input
scaled_X = scaler.transform(X)

# Dự đoán
y_pred = model.predict(scaled_X)

print("Predictions:", y_pred)

# Nếu có ground truth thì tính R²
if y is not None:
    y_true = pd.to_numeric(y, errors="coerce")
    r2 = r2_score(y_true, y_pred)
    print("R²:", r2)
