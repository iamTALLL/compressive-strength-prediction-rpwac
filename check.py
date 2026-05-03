import joblib

model = joblib.load('xgb_woa_best_model.pkl')
print(model.feature_names_in_.tolist())