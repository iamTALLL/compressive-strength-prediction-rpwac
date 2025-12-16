from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import signal
import sys
import os
import subprocess
import platform
import pandas as pd
import json
import datetime
import matplotlib

app = Flask(__name__, template_folder='templates')
LOG_DIR = 'logs'

def list_logs():
    files = [f for f in os.listdir(LOG_DIR) if f.endswith('.csv')]
    files.sort()  # sort theo tên hoặc timestamp
    return files



# Load scaler and model from notebook directory
try:
    scaler = joblib.load('scaler.pkl')
    model = joblib.load('xgb_woa_best_model.pkl')
except FileNotFoundError as e:
    print(f"Error: Model or scaler file not found: {e}")
    sys.exit(1)

def free_port(port):
    """Check and free the specified port if it's in use."""
    system = platform.system()
    try:
        if system in ["Linux", "Darwin"]:  # macOS is 'Darwin'
            # Find PID using lsof
            result = subprocess.run(['lsof', '-ti', f':{port}'], capture_output=True, text=True)
            if result.stdout:
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    if pid:  # Skip empty lines
                        print(f"Found process {pid} using port {port}. Terminating...")
                        subprocess.run(['kill', '-9', pid], check=True)
                        print(f"Process {pid} terminated.")
        elif system == "Windows":
            # Find PID using netstat
            result = subprocess.run(['netstat', '-aon'], capture_output=True, text=True)
            for line in result.stdout.splitlines():
                if f':{port}' in line and 'LISTENING' in line:
                    pid = line.split()[-1]
                    print(f"Found process {pid} using port {port}. Terminating...")
                    subprocess.run(['taskkill', '/PID', pid, '/F'], check=True)
                    print(f"Process {pid} terminated.")
    except subprocess.CalledProcessError as e:
        print(f"Error freeing port {port}: {e}")
    except Exception as e:
        print(f"Unexpected error while freeing port {port}: {e}")

@app.route('/')
@app.route('/home')
def home():
    histogram_dir = os.path.join("static", "histograms")
    histograms = [
        fname
        for fname in os.listdir(histogram_dir)
        if fname.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    return render_template('home.html', histograms=histograms)





@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get 14 parameters from form (matching the first 14 columns in Data1.csv)
            inputs = [
                float(request.form['cement_dosage']),
                float(request.form['cement_compressive_strength']),
                float(request.form['cement_specific_gravity']),
                float(request.form['fine_aggregate_quantity']),
                float(request.form['fine_aggregate_specific_gravity']),
                float(request.form['coarse_aggregate_quantity']),
                float(request.form['coarse_aggregate_specific_gravity']),
                float(request.form['water']),
                float(request.form['water_cement_ratio']),
                float(request.form['plastic_quantity']),
                float(request.form['plastic_tensile_strength']),
                float(request.form['plastic_specific_gravity']),
                float(request.form['slump']),
                float(request.form['concrete_specific_gravity'])
            ]
            
            # Check for negative values
            if any(x < 0 for x in inputs):
                error = "Please enter non-negative values."
                return render_template('predict.html', prediction=None, error=error)
            
            # Reshape and scale data
            input_array = np.array(inputs).reshape(1, -1)
            scaled_input = scaler.transform(input_array)
            
            # Predict c_cs (Concrete compressive strength)
            prediction = model.predict(scaled_input)[0]
            
            df_log = pd.DataFrame([inputs + [prediction]],
                                  columns=[
                                      'cement_dosage', 'cement_compressive_strength', 'cement_specific_gravity',
                                      'fine_aggregate_quantity', 'fine_aggregate_specific_gravity',
                                      'coarse_aggregate_quantity', 'coarse_aggregate_specific_gravity',
                                      'water', 'water_cement_ratio', 'plastic_quantity',
                                      'plastic_tensile_strength', 'plastic_specific_gravity',
                                      'slump', 'concrete_specific_gravity', 'Predicted_c_cs'
                                  ])
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            log_filename = f'single_{timestamp}.csv'
            df_log.to_csv(os.path.join(LOG_DIR, log_filename), index=False)
            
            return render_template('predict.html', prediction=prediction, error=None)
        
        except ValueError:
            error = "Please enter valid numbers for all fields."
            return render_template('predict.html', prediction=None, error=error)
    
    return render_template('predict.html', prediction=None, error=None)

# New route for batch CSV prediction
import time  # nếu chưa import

@app.route('/predict_csv', methods=['GET', 'POST'])
def predict_csv():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('predict_csv.html', predictions=None, error="No file uploaded.")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('predict_csv.html', predictions=None, error="No file selected.")
        
        try:
            df = pd.read_csv(file)
            X = df.values
            scaled_X = scaler.transform(X)
            predictions = model.predict(scaled_X)
            df['Predicted_c_cs'] = predictions

            # --- Lưu log ---
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            log_filename = f'batch_{timestamp}.csv'
            df.to_csv(os.path.join(LOG_DIR, log_filename), index=False)

            return render_template('predict_csv.html', predictions=df.to_html(index=False), error=None)
        except Exception as e:
            return render_template('predict_csv.html', predictions=None, error=f"Error processing file: {e}")
    
    df_html = df.to_html(classes="table table-bordered csv-results", index=False)
    return render_template('predict_csv.html', predictions=df_html, error=None)


@app.route('/chart', methods=['GET'])
def chart():
    logs = list_logs()
    return render_template('chart.html', logs=logs)

# Lấy dữ liệu prediction cho 1 log
@app.route('/chart_data/<log_file>', methods=['GET'])
def chart_data(log_file):
    path = os.path.join(LOG_DIR, log_file)
    if not os.path.exists(path):
        return jsonify({'error': 'File not found'}), 404
    df = pd.read_csv(path)
    # Kỳ vọng: input features..., predicted, optional y_true
    data = df.to_dict(orient='records')
    return jsonify(data)

# Upload reality output để so sánh
@app.route('/upload_reality/<log_file>', methods=['POST'])
def upload_reality(log_file):
    file = request.files.get('reality_csv')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400
    path = os.path.join(LOG_DIR, log_file)
    df_pred = pd.read_csv(path)
    df_actual = pd.read_csv(file)
    # Nối theo index
    if len(df_pred) != len(df_actual):
        return jsonify({'error': 'Length mismatch'}), 400
    df_pred['y_true'] = df_actual.iloc[:, -1]  # giả sử cột cuối là reality
    df_pred.to_csv(path, index=False)
    return jsonify({'success': True})



def signal_handler(sig, frame):
    print('Shutting down server... (Ctrl+C detected)')
    sys.exit(0)

if __name__ == '__main__':
    port = 5004
    signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C
    try:
        print(f"Checking if port {port} is in use...")
        free_port(port)  # Free port before starting server
        print(f"Starting server on port {port}...")
        app.run(debug=True, port=port, threaded=True, use_reloader=False)  # Tắt reloader để tránh kill
    except KeyboardInterrupt:
        print('Server stopped.')
    except Exception as e:
        print(f'Error: {e}. Attempting to free port {port}...')
        free_port(port)
        sys.exit(1)