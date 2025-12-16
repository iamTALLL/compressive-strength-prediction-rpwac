from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import sys
import os
import pandas as pd
import datetime
import logging
from logging.handlers import RotatingFileHandler

try:
    from mealpy.evolutionary_based.GA import BaseGA 
    from mealpy.swarm_based.FOX import OriginalFOX # Import lớp giải thuật chính xác    
    from mealpy.utils.problem import Problem
    from mealpy.utils.space import FloatVar
except ImportError:
    print("WARNING: mealpy not found. Inverse Design feature will be disabled.")
    # Cung cấp một biến cờ để vô hiệu hóa tính năng nếu không có mealpy
    OPTIMIZE_ENABLED = False
else:
    OPTIMIZE_ENABLED = True

# --- CONFIGURATION ---
LOG_DIR = 'logs'
PORT = 5004 # Sử dụng cổng mặc định của ứng dụng

# Thiết lập Logging chuẩn Python (Ghi log hệ thống)
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

handler = RotatingFileHandler(
    os.path.join(LOG_DIR, 'app_system.log'), 
    maxBytes=100000, 
    backupCount=3
)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
handler.setFormatter(formatter)

app = Flask(__name__, template_folder='templates')
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)


# Load scaler and model from notebook directory
scaler = None
model = None
MODEL_FEATURES = [
    'cement_dosage', 'cement_compressive_strength', 'cement_specific_gravity',
    'fine_aggregate_quantity', 'fine_aggregate_specific_gravity',
    'coarse_aggregate_quantity', 'coarse_aggregate_specific_gravity',
    'water', 'water_cement_ratio', 'plastic_quantity',
    'plastic_tensile_strength', 'plastic_specific_gravity',
    'slump', 'concrete_specific_gravity'
]

try:
    scaler = joblib.load('scaler.pkl')
    model = joblib.load('xgb_woa_best_model.pkl')
    app.logger.info("Model and scaler loaded successfully.")

    # ==== 2. Feature list (Lấy từ mô hình hoặc mặc định) ====
    try:
        # Nếu mô hình có feature_names_in_ (như XGBoost)
        all_features = model.feature_names_in_.tolist() 
    except AttributeError:
        # Danh sách 14 features (Đã chuẩn hóa tên trong mã Gradio)
        all_features = [
            "c_d", "ce_cs", "ce_sg", "f_q", "f_sg", "c_q", "ca_sg", 
            "w", "w/c", "p_q", "p_ts", "p_sg", "slump", "c_sg"
        ]

    # Danh sách tên đầy đủ
    full_names = [
        'Cement dosage', 'Cement compressive strength', 'Cement specific gravity',
        'Fine aggregate quantity', 'Fine aggregate specific gravity',
        'Coarse aggregate quantity', 'Corse aggregate specific gravity',
        'Water', 'Water/cement', 'Plastic quantity',
        'Plastic tensile strength', 'Plastic specific gravity',
        'Slump', 'Concrete specific gravity'
    ]
    name_map = dict(zip(all_features, full_names))

    # ==== 3. Sinh feature bounds từ scaler ====
    if hasattr(scaler, "data_min_") and hasattr(scaler, "data_max_"):
        # Trường hợp MinMaxScaler
        feature_bounds = {
            feat: (float(scaler.data_min_[i]), float(scaler.data_max_[i]))
            for i, feat in enumerate(all_features)
        }
    elif hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        # Trường hợp StandardScaler: dùng mean ± 3*std làm bounds
        feature_bounds = {
            feat: (float(scaler.mean_[i] - 3*scaler.scale_[i]),
                   float(scaler.mean_[i] + 3*scaler.scale_[i]))
            for i, feat in enumerate(all_features)
        }
    else:
        raise ValueError("Scaler is not supported. Please use MinMaxScaler or StandardScaler.")
    app.logger.info("Feature bounds created.")

except FileNotFoundError as e:
    app.logger.critical(f"Error: Model or scaler file not found: {e}. Exiting.", exc_info=True)
    sys.exit(1)
except Exception as e:
    app.logger.critical(f"FATAL ERROR during model loading or initialization: {e}", exc_info=True)
    sys.exit(1)

if OPTIMIZE_ENABLED:
    # Các hằng số từ mã Gradio
    PENALTY_FACTOR = 50000 
    V_PLASTIC_ABSOLUTE_MAX = 0.20
    V_MIN_TARGET = 0.95
    V_MAX_TARGET = 1.05
    L1 = 0.005 # Tối thiểu Xi măng
    L2 = 0.0025 # Tối đa Nhựa

    def optimize_materials(target_CS, **fixed_features):
        """Hàm tối ưu hóa thành phần vật liệu sử dụng thuật toán FOX (BaseGA)."""
        
        fixed_idx = [all_features.index(k) for k in fixed_features.keys()]
        fixed_values = [float(v) for v in fixed_features.values()]
        free_idx = [i for i in range(len(all_features)) if i not in fixed_idx]

        if not free_idx:
            # Nếu tất cả các biến đã được cố định (không có biến tự do)
            return {
                "Error": "All 14 components are fixed. No optimization is possible. Please unfix at least one component.",
                "Predicted_CS": None
            }

        lb = [feature_bounds[all_features[i]][0] for i in free_idx]
        ub = [feature_bounds[all_features[i]][1] for i in free_idx]
        variables = [FloatVar(lb=lb[i], ub=ub[i]) for i in range(len(free_idx))]

        class MyProblem(Problem):
            def __init__(self):
                super().__init__(
                    obj_func=self.fitness,
                    n_dim=len(free_idx),
                    bounds=variables
                )

            def fitness(self, candidate):
                # Khôi phục vector đầu vào đầy đủ (14 chiều)
                full_vector = [0]*len(all_features)
                for i, idx in enumerate(fixed_idx):
                    full_vector[idx] = fixed_values[i]
                for i, idx in enumerate(free_idx):
                    full_vector[idx] = candidate[i]
                
                feature_map = dict(zip(all_features, full_vector))
                
                # Lấy các khối lượng (kg/m3)
                c_d_mass = feature_map.get('c_d', 0)
                p_q_mass = feature_map.get('p_q', 0)
                f_q_mass = feature_map.get('f_q', 0)
                c_q_mass = feature_map.get('c_q', 0)
                w_mass = feature_map.get('w', 0)
                
                # Lấy và XỬ LÝ các giá trị Tỷ trọng (SG)
                ce_sg = feature_map.get('ce_sg', 1.0)
                f_sg = feature_map.get('f_sg', 1.0)
                ca_sg = feature_map.get('ca_sg', 1.0)
                p_sg = feature_map.get('p_sg', 1.0)

                # Đảm bảo không có mẫu số nào bằng 0
                ce_sg = ce_sg if ce_sg != 0 else 1.0
                f_sg = f_sg if f_sg != 0 else 1.0
                ca_sg = ca_sg if ca_sg != 0 else 1.0
                p_sg = p_sg if p_sg != 0 else 1.0
                
                # Tính thể tích từng thành phần (V = M / (SG * 1000) - TÍNH VỚI ĐƠN VỊ m3)
                V_cement = c_d_mass / (ce_sg * 1000)
                V_fine_agg = f_q_mass / (f_sg * 1000)
                V_coarse_agg = c_q_mass / (ca_sg * 1000)
                V_plastic = p_q_mass / (p_sg * 1000)
                V_water = w_mass / 1000 # RHO_WATER = 1000

                V_total = V_cement + V_fine_agg + V_coarse_agg + V_plastic + V_water

                # --- TÍNH HÌNH PHẠT (Penalty Logic) ---
                penalty_total_volume = 0
                # Ràng buộc: Thể tích Khối bê tông (1m3)
                if V_total < V_MIN_TARGET:
                    violation = V_MIN_TARGET - V_total
                    penalty_total_volume = PENALTY_FACTOR * violation
                elif V_total > V_MAX_TARGET:
                    violation = V_total - V_MAX_TARGET
                    penalty_total_volume = PENALTY_FACTOR * violation
                    
                penalty_plastic_volume = 0
                # Ràng buộc: Thể tích nhựa không quá 20%
                if V_plastic > V_PLASTIC_ABSOLUTE_MAX:
                    violation = V_plastic - V_PLASTIC_ABSOLUTE_MAX
                    # Penalty cao hơn (10x) vì đây là ràng buộc cứng quan trọng
                    penalty_plastic_volume = PENALTY_FACTOR * 10 * violation
                
                # 2. Dự đoán Cường độ Nén (CS)
                x_scaled = scaler.transform(np.array([full_vector])) 
                pred_CS = model.predict(x_scaled)[0]
                delta_CS = abs(pred_CS - target_CS) # Mục tiêu 1: CS = Target CS
                
                # 3. Kết hợp Đa Mục tiêu
                total_fitness = (
                    delta_CS + 
                    (L1 * c_d_mass) - # Mục tiêu 2: Tối thiểu Xi măng (c_d_mass)
                    (L2 * p_q_mass) + # Mục tiêu 3: Tối đa Nhựa (p_q_mass)
                    penalty_total_volume + 
                    penalty_plastic_volume
                )
                return total_fitness
        
        problem = MyProblem()
        ga_solver = OriginalFOX(epoch=100, pop_size=30) 
        
        try:
            best_agent = ga_solver.solve(problem)
            best_solution = best_agent.solution
        except Exception as e:
            app.logger.error(f"GA Solver error: {e}", exc_info=True)
            return {
                "Error": f"GA Solver Error: {e}. Check server logs for details.",
                "Predicted_CS": None
            }

        # Khôi phục vector kết quả cuối cùng
        result = [0]*len(all_features)
        for i, idx in enumerate(fixed_idx):
            result[idx] = fixed_values[i]
        for i, idx in enumerate(free_idx):
            result[idx] = best_solution[i]

        # Dự đoán lại CS với nghiệm tìm được
        pred_CS = model.predict(scaler.transform(np.array([result])))[0]
        
        return {
            "Optimized Materials": {feat: round(val, 4) for feat, val in zip(all_features, result)},
            "Predicted_CS": round(pred_CS, 2)
        }


def list_logs():
    files = [f for f in os.listdir(LOG_DIR) if f.endswith('.csv')]
    files.sort()
    return files


@app.route('/')
@app.route('/home')
def home():
    app.logger.info("Accessing home page.")
    histogram_dir = os.path.join("static", "histograms")
    histograms = [
        fname
        for fname in os.listdir(histogram_dir)
        if fname.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    # SỬA LỖI: Cần truyền OPTIMIZE_ENABLED cho home.html (và base.html)
    return render_template('home.html', 
                           histograms=histograms,
                           optimize_enabled=OPTIMIZE_ENABLED)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        app.logger.info("Received POST request for single prediction.")
        try:
            # Get 14 parameters from form
            inputs = [
                float(request.form[feat]) for feat in MODEL_FEATURES
            ]
            
            # Check for negative values
            if any(x < 0 for x in inputs):
                error = "Please enter non-negative values."
                app.logger.warning(f"Validation error: Negative values received: {inputs}")
                return render_template('predict.html', prediction=None, error=error, features=MODEL_FEATURES)
            
            # Reshape and scale data
            input_array = np.array(inputs).reshape(1, -1)
            scaled_input = scaler.transform(input_array)
            
            # Predict c_cs (Concrete compressive strength)
            prediction = model.predict(scaled_input)[0]
            
            # Data Logging
            df_log = pd.DataFrame([inputs + [prediction]], columns=MODEL_FEATURES + ['Predicted_c_cs'])
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            log_filename = f'single_{timestamp}.csv'
            df_log.to_csv(os.path.join(LOG_DIR, log_filename), index=False)
            app.logger.info(f"Single prediction logged successfully to {log_filename}")
            
            return render_template('predict.html', prediction=prediction, error=None, features=MODEL_FEATURES)
            
        except ValueError:
            error = "Please enter valid numbers for all fields."
            app.logger.error("Input parsing failed (ValueError).", exc_info=True)
            return render_template('predict.html', prediction=None, error=error, features=MODEL_FEATURES)
        except Exception as e:
            error = f"An unexpected error occurred: {str(e)}"
            app.logger.critical(f"Critical prediction error: {e}", exc_info=True)
            return render_template('predict.html', prediction=None, error=error, features=MODEL_FEATURES)
    
    return render_template('predict.html', prediction=None, error=None, features=MODEL_FEATURES)


@app.route('/predict_csv', methods=['GET', 'POST'])
def predict_csv():
    # Giữ nguyên logic trả về HTML cho GET và POST (trong trường hợp lỗi)
    if request.method == 'POST':
        app.logger.info("Received POST request for batch CSV prediction.")
        if 'file' not in request.files:
            return render_template('predict_csv.html', predictions=None, error="No file uploaded.")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('predict_csv.html', predictions=None, error="No file selected.")
        
        try:
            df = pd.read_csv(file)
            
            # Thêm kiểm tra số lượng cột
            if df.shape[1] != len(all_features):
                 error = f"CSV file must have {len(all_features)} columns, but found {df.shape[1]}. Expected features: {', '.join(all_features)}"
                 app.logger.error(error)
                 return render_template('predict_csv.html', predictions=None, error=error)
                 
            X = df.values
            scaled_X = scaler.transform(X)
            predictions = model.predict(scaled_X)
            
            # Sử dụng tên cột chuẩn (tên ngắn) từ all_features
            df.columns = all_features
            df['Predicted_c_cs'] = predictions

            # --- Lưu log ---
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            log_filename = f'batch_{timestamp}.csv'
            df.to_csv(os.path.join(LOG_DIR, log_filename), index=False)
            app.logger.info(f"Batch prediction successful and logged to {log_filename}")

            # Trả về HTML table cho frontend
            # Đổi tên cột hiển thị cho dễ hiểu
            display_df = df.rename(columns=name_map).fillna(0).round(4)
            return render_template('predict_csv.html', predictions=display_df.to_html(index=False), error=None)
        
        except Exception as e:
            app.logger.error(f"Error processing batch file: {e}", exc_info=True)
            return render_template('predict_csv.html', predictions=None, error=f"Error processing file: {e}")
    
    df_html = "<p>Upload CSV file for batch prediction.</p>" 
    return render_template('predict_csv.html', predictions=df_html, error=None)


@app.route('/optimize', methods=['GET', 'POST'])
def inverse_design():
    """Endpoint xử lý Inverse Design/Tối ưu hóa."""
    
    # SỬA LỖI: Truyền biến OPTIMIZE_ENABLED và hằng số penalty
    
    # Chuẩn bị biến để truyền cho template (dù có lỗi hay không)
    template_vars = {
        'all_features': all_features,
        'name_map': name_map,
        'optimize_enabled': OPTIMIZE_ENABLED
    }
    
    if OPTIMIZE_ENABLED:
        template_vars.update({
            'L1': L1,
            'L2': L2
        })
    else:
        # Nếu tính năng bị vô hiệu hóa, chỉ trả về template với cảnh báo
        return render_template('optimize.html', 
                               error="Inverse Design is disabled (mealpy not found).", 
                               **template_vars)
        
    if request.method == 'POST':
        app.logger.info("Received POST request for Inverse Design.")
        try:
            # Lấy Target CS
            target_CS = float(request.form.get('target_cs'))
            fixed_features = {}
            
            # Lấy 14 giá trị features từ form
            for feat in all_features:
                val_str = request.form.get(feat)
                if val_str:
                    val = float(val_str)
                    # Chỉ coi là ràng buộc cố định nếu giá trị khác None/0
                    if val != 0:
                        fixed_features[feat] = val
            
            # Kiểm tra Target CS
            if target_CS < 10 or target_CS > 90:
                # Trả về template_vars với lỗi
                return render_template('optimize.html', 
                                       error=f"WARNING: Target CS {target_CS} MPa is outside the range of reliable data (10-90 MPa).", 
                                       **template_vars)

            # Chạy tối ưu hóa
            results = optimize_materials(target_CS, **fixed_features)
            
            if "Error" in results:
                return render_template('optimize.html', error=results["Error"], **template_vars)

            optimized_materials = results["Optimized Materials"]
            predicted_cs = results["Predicted_CS"]
            
            # Định dạng output thành danh sách các cặp (Component, Value)
            optimized_list = [(name_map.get(k, k), v) for k, v in optimized_materials.items()]
            
            # Kết hợp các biến kết quả vào template_vars
            template_vars.update({
                'target_cs': target_CS,
                'predicted_cs': predicted_cs, 
                'optimized_list': optimized_list
            })
            
            return render_template('optimize.html', **template_vars)

        except ValueError:
            error = "Please enter valid numbers for Target CS and all fixed fields."
            app.logger.error("Input parsing failed (ValueError) in /optimize.", exc_info=True)
            return render_template('optimize.html', error=error, **template_vars)
        except Exception as e:
            error = f"An unexpected error occurred during optimization: {str(e)}"
            app.logger.critical(f"Critical optimization error: {e}", exc_info=True)
            return render_template('optimize.html', error=error, **template_vars)
    
    # Xử lý GET request cho /optimize
    return render_template('optimize.html', **template_vars)


@app.route('/chart', methods=['GET'])
def chart():
    logs = list_logs()
    return render_template('chart.html', logs=logs)

# Lấy dữ liệu prediction cho 1 log (API - nên trả về JSON)
@app.route('/chart_data/<log_file>', methods=['GET'])
def chart_data(log_file):
    path = os.path.join(LOG_DIR, log_file)
    if not os.path.exists(path):
        app.logger.warning(f"Chart data request for non-existent file: {log_file}")
        return jsonify({'error': 'File not found'}), 404
    df = pd.read_csv(path)
    # Đổi tên cột cho dễ hiểu khi hiển thị trên biểu đồ
    df = df.rename(columns=name_map)
    data = df.to_dict(orient='records')
    return jsonify(data)

# Upload reality output để so sánh (API - nên trả về JSON)
@app.route('/upload_reality/<log_file>', methods=['POST'])
def upload_reality(log_file):
    file = request.files.get('reality_csv')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400
        
    path = os.path.join(LOG_DIR, log_file)
    try:
        df_pred = pd.read_csv(path)
        df_actual = pd.read_csv(file)
    except FileNotFoundError:
        app.logger.error(f"Attempted reality upload failed: Log file {log_file} not found.")
        return jsonify({'error': 'Target log file not found'}), 404
    except Exception as e:
        app.logger.error(f"Failed to read CSV files for reality upload: {e}")
        return jsonify({'error': 'Failed to read uploaded CSV data'}), 400

    if len(df_pred) != len(df_actual):
        app.logger.warning("Reality upload mismatch: Prediction length != Actual length.")
        return jsonify({'error': 'Length mismatch'}), 400
    
    if df_actual.shape[1] < 1:
        return jsonify({'error': 'Reality CSV must contain at least one column (the actual values).'}), 400

    df_pred['y_true'] = df_actual.iloc[:, -1] # giả sử cột cuối là reality
    df_pred.to_csv(path, index=False)
    app.logger.info(f"Reality data uploaded and saved for log file: {log_file}")
    return jsonify({'success': True})


if __name__ == '__main__':
    app.logger.info(f"Starting server on port {PORT}...")
    # Tắt use_reloader và debug để chạy ổn định hơn trên server/Docker
    app.run(host='0.0.0.0', port=PORT, debug=False, use_reloader=False)