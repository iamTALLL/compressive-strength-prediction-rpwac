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