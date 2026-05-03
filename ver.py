import numpy
import pandas
import joblib
import gunicorn
import sys

print(f"Python Version: {sys.version.split()[0]}")
print(f"NumPy Version: {numpy.__version__}")
print(f"Pandas Version: {pandas.__version__}")
print(f"Joblib Version: {joblib.__version__}")
print(f"Gunicorn Version: {gunicorn.__version__}")

import joblib
model = joblib.load('xgb_woa_best_model.pkl')
print(model.feature_names_in_.tolist())