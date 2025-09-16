import joblib
import pandas as pd
from sklearn.metrics import r2_score

# Load model
model = joblib.load("xgb_woa_best_model.pkl")

# Đọc dữ liệu
df = pd.read_csv("Data1.csv")

print(df.columns)              # dạng Index([...])
print(list(df.columns)) 
print(df.dtypes)

print("Data shape:", df.shape)
print("Data columns:", df.columns.tolist())