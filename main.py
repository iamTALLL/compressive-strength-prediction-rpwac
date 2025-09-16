from flask import Flask, request, render_template_string
import joblib
import pandas as pd
import os
import sys

app = Flask(__name__)

# Load scaler and model from notebook directory
try:
    scaler = joblib.load('scaler.pkl')
    model = joblib.load('xgb_woa_best_model.pkl')
except FileNotFoundError as e:
    print(f"Error: Model or scaler file not found: {e}")
    sys.exit(1)

MODEL_PATH = "xgb_woa_best_model.pkl"
MODEL = None
if os.path.exists(MODEL_PATH):
    MODEL = joblib.load(MODEL_PATH)
    print("Loaded best model from", MODEL_PATH)
else:
    print("Warning: Model file not found.")

# Danh sách feature cố định
FEATURE_NAMES = [
    "c_d", "ce_cs", "ce_sg", "f_q", "f_sg",
    "c_q", "ca_sg", "w", "w/c", "p_q",
    "p_ts", "p_sg", "slump", "c_sg"
]

HTML = """
<!doctype html>
<html lang="vi">
  <head>
    <meta charset="utf-8">
    <title>XGB-WOA Prediction Demo</title>
    <style>
      body{font-family:Arial, sans-serif;max-width:800px;margin:30px auto;padding:10px}
      h2{color:#333}
      label{display:block;margin-top:10px}
      input[type=text]{width:100%;padding:6px;margin-top:4px}
      button{margin-top:10px;padding:8px 12px}
      .result{margin-top:18px;padding:12px;border:1px solid #ccc;background:#f9f9f9}
      .grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
    </style>
  </head>
  <body>
    <h2>Dự đoán với model XGB-WOA</h2>
    <form method="post" action="/predict_one">
      <p><strong>Nhập dữ liệu 1 record</strong></p>
      <div class="grid">
        {% for f in features %}
          <div>
            <label for="{{f}}">{{f}}</label>
            <input id="{{f}}" name="{{f}}" type="text" required>
          </div>
        {% endfor %}
      </div>
      <button type="submit">Predict One</button>
    </form>

    <hr>
    <form method="post" action="/predict_csv" enctype="multipart/form-data">
      <p><strong>Upload CSV nhiều record</strong></p>
      <input type="file" name="file" accept=".csv" required>
      <button type="submit">Predict CSV</button>
    </form>

    {% if prediction is defined %}
      <div class="result">
        <strong>Prediction:</strong> {{ prediction }}
      </div>
    {% endif %}

    {% if predictions is defined %}
      <div class="result">
        <h4>Kết quả nhiều record:</h4>
        <pre>{{ predictions }}</pre>
      </div>
    {% endif %}
  </body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML, features=FEATURE_NAMES)

@app.route("/predict_one", methods=["POST"])
def predict_one():
    try:
        # Lấy dữ liệu từ form
        values = []
        for f in FEATURE_NAMES:
            val = request.form.get(f, "").strip()
            values.append(float(val))
        df_input = pd.DataFrame([values], columns=FEATURE_NAMES)

        pred = MODEL.predict(df_input) if MODEL else ["No model"]
        prediction = str(pred[0])
    except Exception as e:
        prediction = f"Error: {e}"
    return render_template_string(HTML, features=FEATURE_NAMES, prediction=prediction)

@app.route("/predict_csv", methods=["POST"])
def predict_csv():
    file = request.files.get("file")
    if not file:
        return "No file uploaded", 400
    try:
        df = pd.read_csv(file)
        preds = MODEL.predict(df) if MODEL else ["No model"] * len(df)
        df["Prediction"] = preds
        preview = df.head(10).to_string(index=False)
        return render_template_string(HTML, features=FEATURE_NAMES, predictions=preview)
    except Exception as e:
        return f"Error reading CSV or predicting: {e}", 500

if __name__ == "__main__":
    app.run(debug=True)
