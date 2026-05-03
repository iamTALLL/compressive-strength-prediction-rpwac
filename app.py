from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import sys
import os
import pandas as pd
import datetime
import logging
import warnings
warnings.filterwarnings('ignore')
from logging.handlers import RotatingFileHandler

try:
    from mealpy.swarm_based.FOX import OriginalFOX
    from mealpy.utils.space import FloatVar
except ImportError:
    print("WARNING: mealpy not found. Inverse Design feature will be disabled.")
    OPTIMIZE_ENABLED = False
else:
    OPTIMIZE_ENABLED = True

# ============================================================
# CONFIGURATION
# ============================================================
LOG_DIR = 'logs'
PORT    = 5004

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

handler = RotatingFileHandler(
    os.path.join(LOG_DIR, 'app_system.log'),
    maxBytes=100000,
    backupCount=3
)
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s'))

app = Flask(__name__, template_folder='templates')
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)

# ============================================================
# MODEL & SCALER LOADING
# ============================================================
scaler = None
model  = None

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
    model  = joblib.load('xgb_woa_best_model.pkl')
    app.logger.info("Model and scaler loaded successfully.")

    try:
        all_features = model.feature_names_in_.tolist()
    except AttributeError:
        all_features = [
            "c_d", "ce_cs", "ce_sg", "f_q", "f_sg", "c_q", "c_sg",
            "w", "w/c", "p_q", "p_ts", "p_sg", "slump", "c_sg1"
        ]

    full_names = [
        'Cement dosage', 'Cement compressive strength', 'Cement specific gravity',
        'Fine aggregate quantity', 'Fine aggregate specific gravity',
        'Coarse aggregate quantity', 'Coarse aggregate specific gravity',
        'Water', 'Water/cement', 'Plastic quantity',
        'Plastic tensile strength', 'Plastic specific gravity',
        'Slump', 'Concrete specific gravity'
    ]
    name_map = dict(zip(all_features, full_names))

    if hasattr(scaler, "data_min_") and hasattr(scaler, "data_max_"):
        feature_bounds = {
            feat: (float(scaler.data_min_[i]), float(scaler.data_max_[i]))
            for i, feat in enumerate(all_features)
        }
    elif hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        feature_bounds = {
            feat: (float(scaler.mean_[i] - 3 * scaler.scale_[i]),
                   float(scaler.mean_[i] + 3 * scaler.scale_[i]))
            for i, feat in enumerate(all_features)
        }
    else:
        raise ValueError("Unsupported scaler type. Use MinMaxScaler or StandardScaler.")

    app.logger.info("Feature bounds created successfully.")

except FileNotFoundError as e:
    app.logger.critical(f"Model/scaler file not found: {e}")
    sys.exit(1)
except Exception as e:
    app.logger.critical(f"FATAL ERROR during initialization: {e}")
    sys.exit(1)


# ============================================================
# HELPER — dùng DataFrame để tránh warning feature names
# ============================================================
def predict_cs(full_vector):
    """Predict CS từ full_vector (list 14 giá trị), trả về float."""
    df_input = pd.DataFrame([full_vector], columns=all_features)
    scaled   = scaler.transform(df_input)
    return float(model.predict(scaled)[0])


# ============================================================
# OPTIMIZATION CONFIG
# ============================================================
if OPTIMIZE_ENABLED:
    PENALTY_FACTOR         = 5     # Giảm mạnh xuống 5
    V_MIN_TARGET           = 0.90
    V_MAX_TARGET           = 1.10
    L1                     = 0.005
    L2                     = 0.0025
    MAX_RETRIES            = 3
    EPOCH_BASE             = 200
    EPOCH_RETRY            = 400
    POP_SIZE               = 60

    def optimize_materials(target_CS, **fixed_features):
        fixed_idx    = [all_features.index(k) for k in fixed_features.keys()]
        fixed_values = [float(v) for v in fixed_features.values()]
        free_idx     = [i for i in range(len(all_features)) if i not in fixed_idx]

        if not free_idx:
            return {
                "Error": "All 14 components are fixed. Please unfix at least one.",
                "Predicted_CS": None
            }

        lb = [feature_bounds[all_features[i]][0] for i in free_idx]
        ub = [feature_bounds[all_features[i]][1] for i in free_idx]

        # ── Fitness function ────────────────────────────────────
        def fitness_func(x):
            # Khôi phục full vector 14 chiều
            full_vector = [0.0] * len(all_features)
            for i, idx in enumerate(fixed_idx):
                full_vector[idx] = fixed_values[i]
            for i, idx in enumerate(free_idx):
                full_vector[idx] = x[i]

            fm = dict(zip(all_features, full_vector))

            # Khối lượng
            c_d_mass = fm.get('c_d',    0.0)
            p_q_mass = fm.get('p_q',    0.0)
            f_q_mass = fm.get('f_q',    0.0)
            c_q_mass = fm.get('c_q',    0.0)
            w_mass   = fm.get('w',      0.0)

            # Tỷ trọng (tránh chia 0)
            ce_sg = fm.get('ce_sg', 1.0) or 1.0
            f_sg  = fm.get('f_sg',  1.0) or 1.0
            c_sg1 = fm.get('c_sg1', 1.0) or 1.0
            p_sg  = fm.get('p_sg',  1.0) or 1.0

            # Thể tích (m3)
            V_cement     = c_d_mass / (ce_sg * 1000)
            V_fine_agg   = f_q_mass / (f_sg  * 1000)
            V_coarse_agg = c_q_mass / (c_sg1 * 1000)
            V_plastic    = p_q_mass / (p_sg  * 1000)
            V_water      = w_mass   / 1000
            V_total      = V_cement + V_fine_agg + V_coarse_agg + V_plastic + V_water

            # Dự đoán CS — dùng DataFrame tránh warning
            pred_CS = predict_cs(full_vector)

            # Penalty thể tích tổng (phải ~1m3)
            penalty_vol = 0.0
            if V_total < V_MIN_TARGET:
                penalty_vol = PENALTY_FACTOR * (V_MIN_TARGET - V_total)
            elif V_total > V_MAX_TARGET:
                penalty_vol = PENALTY_FACTOR * (V_total - V_MAX_TARGET)

            # Penalty nhựa (không quá 20%)
            penalty_plastic = 0.0
            if V_plastic > V_PLASTIC_ABSOLUTE_MAX:
                penalty_plastic = PENALTY_FACTOR * 5 * (V_plastic - V_PLASTIC_ABSOLUTE_MAX)

            # Penalty CS thấp hơn target
            penalty_cs = 0.0
            if pred_CS < target_CS:
                penalty_cs = PENALTY_FACTOR * (target_CS - pred_CS)

            # Overshoot — minimize nhưng chỉ tính khi pred >= target
            delta_CS = (pred_CS - target_CS) if pred_CS >= target_CS else 0.0

            fitness = (
                delta_CS
                + L1 * c_d_mass      # Minimize xi măng
                - L2 * p_q_mass      # Maximize nhựa
                + penalty_vol
                + penalty_plastic
                + penalty_cs         # Đảm bảo pred >= target
            )
            return fitness  # FOX minimize

        # ── Retry loop ──────────────────────────────────────────
        best_solution = None
        best_pred_CS  = -np.inf
        best_delta    = np.inf

        for attempt in range(1, MAX_RETRIES + 1):
            epoch = EPOCH_BASE if attempt == 1 else EPOCH_RETRY
            app.logger.info(f"Optimization attempt {attempt}/{MAX_RETRIES}, epoch={epoch}, pop={POP_SIZE}")

            problem_dict = {
                "obj_func": fitness_func,
                "bounds":   FloatVar(lb=lb, ub=ub),
                "minmax":   "min",
            }

            try:
                solver = OriginalFOX(epoch=epoch, pop_size=POP_SIZE)
                agent  = solver.solve(problem_dict)
                sol    = agent.solution

                # Tính pred_CS cho solution này
                full_vector = [0.0] * len(all_features)
                for i, idx in enumerate(fixed_idx):
                    full_vector[idx] = fixed_values[i]
                for i, idx in enumerate(free_idx):
                    full_vector[idx] = sol[i]

                pred_CS = predict_cs(full_vector)
                delta   = pred_CS - target_CS

                app.logger.info(
                    f"  Attempt {attempt}: pred_CS={pred_CS:.2f}, "
                    f"target={target_CS}, delta={delta:.2f}"
                )

                # Ưu tiên: pred >= target VÀ delta nhỏ nhất
                if pred_CS >= target_CS and delta < best_delta:
                    best_delta    = delta
                    best_solution = full_vector[:]
                    best_pred_CS  = pred_CS
                    app.logger.info(f"  ✅ New best: pred={pred_CS:.2f}, delta={delta:.2f}")

                # Early stop nếu đủ gần
                if pred_CS >= target_CS and delta < 2.0:
                    app.logger.info(f"  Early stop: delta={delta:.2f} < 2.0 MPa")
                    break

            except Exception as e:
                app.logger.error(f"Attempt {attempt} failed: {e}", exc_info=True)
                continue

        # Nếu không attempt nào đạt target → chạy thêm lần cuối mạnh hơn
        if best_solution is None:
            app.logger.warning("No attempt reached target CS. Running final heavy pass...")
            problem_dict = {
                "obj_func": fitness_func,
                "bounds":   FloatVar(lb=lb, ub=ub),
                "minmax":   "min",
            }
            try:
                solver = OriginalFOX(epoch=500, pop_size=80)
                agent  = solver.solve(problem_dict)
                sol    = agent.solution

                full_vector = [0.0] * len(all_features)
                for i, idx in enumerate(fixed_idx):
                    full_vector[idx] = fixed_values[i]
                for i, idx in enumerate(free_idx):
                    full_vector[idx] = sol[i]

                best_pred_CS  = predict_cs(full_vector)
                best_solution = full_vector[:]
                best_delta    = best_pred_CS - target_CS
                app.logger.info(f"  Final pass: pred_CS={best_pred_CS:.2f}, delta={best_delta:.2f}")

            except Exception as e:
                return {"Error": f"All optimization attempts failed: {e}", "Predicted_CS": None}

        return {
            "Optimized Materials": {
                feat: round(val, 4)
                for feat, val in zip(all_features, best_solution)
            },
            "Predicted_CS": round(best_pred_CS, 2),
            "Delta_CS":     round(best_pred_CS - target_CS, 2),
        }


# ============================================================
# UTILITIES
# ============================================================
def list_logs():
    files = [f for f in os.listdir(LOG_DIR) if f.endswith('.csv')]
    files.sort()
    return files


# ============================================================
# ROUTES
# ============================================================
@app.route('/')
@app.route('/home')
def home():
    app.logger.info("Accessing home page.")
    histogram_dir = os.path.join("static", "histograms")
    histograms = [
        fname for fname in os.listdir(histogram_dir)
        if fname.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    return render_template('home.html',
                           histograms=histograms,
                           optimize_enabled=OPTIMIZE_ENABLED)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        app.logger.info("Received POST request for single prediction.")
        try:
            inputs = [float(request.form[feat]) for feat in MODEL_FEATURES]

            if any(x < 0 for x in inputs):
                error = "Please enter non-negative values."
                return render_template('predict.html',
                                       prediction=None, error=error,
                                       features=MODEL_FEATURES)

            # Dùng DataFrame tránh warning feature names
            df_input   = pd.DataFrame([inputs], columns=MODEL_FEATURES)
            scaled     = scaler.transform(df_input)
            prediction = float(model.predict(scaled)[0])

            # Log
            df_log = pd.DataFrame([inputs + [prediction]],
                                  columns=MODEL_FEATURES + ['Predicted_c_cs'])
            timestamp    = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            log_filename = f'single_{timestamp}.csv'
            df_log.to_csv(os.path.join(LOG_DIR, log_filename), index=False)
            app.logger.info(f"Single prediction logged to {log_filename}")

            return render_template('predict.html',
                                   prediction=prediction, error=None,
                                   features=MODEL_FEATURES)

        except ValueError:
            return render_template('predict.html',
                                   prediction=None,
                                   error="Please enter valid numbers for all fields.",
                                   features=MODEL_FEATURES)
        except Exception as e:
            app.logger.critical(f"Critical prediction error: {e}", exc_info=True)
            return render_template('predict.html',
                                   prediction=None,
                                   error=f"An unexpected error occurred: {str(e)}",
                                   features=MODEL_FEATURES)

    return render_template('predict.html',
                           prediction=None, error=None,
                           features=MODEL_FEATURES)


@app.route('/predict_csv', methods=['GET', 'POST'])
def predict_csv():
    if request.method == 'POST':
        app.logger.info("Received POST for batch CSV prediction.")
        if 'file' not in request.files:
            return render_template('predict_csv.html',
                                   predictions=None, error="No file uploaded.")

        file = request.files['file']
        if file.filename == '':
            return render_template('predict_csv.html',
                                   predictions=None, error="No file selected.")

        try:
            df = pd.read_csv(file)

            if df.shape[1] != len(all_features):
                error = (f"CSV must have {len(all_features)} columns, "
                         f"found {df.shape[1]}. "
                         f"Expected: {', '.join(all_features)}")
                return render_template('predict_csv.html',
                                       predictions=None, error=error)

            df.columns    = all_features
            scaled_X      = scaler.transform(df)   # df đã có tên cột → không warning
            predictions   = model.predict(scaled_X)
            df['Predicted_c_cs'] = predictions

            timestamp    = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            log_filename = f'batch_{timestamp}.csv'
            df.to_csv(os.path.join(LOG_DIR, log_filename), index=False)
            app.logger.info(f"Batch prediction logged to {log_filename}")

            display_df = df.rename(columns=name_map).fillna(0).round(4)
            return render_template('predict_csv.html',
                                   predictions=display_df.to_html(index=False),
                                   error=None)

        except Exception as e:
            app.logger.error(f"Batch prediction error: {e}", exc_info=True)
            return render_template('predict_csv.html',
                                   predictions=None,
                                   error=f"Error processing file: {e}")

    return render_template('predict_csv.html',
                           predictions="<p>Upload CSV file for batch prediction.</p>",
                           error=None)


@app.route('/optimize', methods=['GET', 'POST'])
def inverse_design():
    template_vars = {
        'all_features':     all_features,
        'name_map':         name_map,
        'optimize_enabled': OPTIMIZE_ENABLED,
    }

    if not OPTIMIZE_ENABLED:
        return render_template('optimize.html',
                               error="Inverse Design is disabled (mealpy not found).",
                               **template_vars)

    template_vars.update({'L1': L1, 'L2': L2})

    if request.method == 'POST':
        app.logger.info("Received POST for Inverse Design.")
        try:
            target_CS      = float(request.form.get('target_cs'))
            fixed_features = {}

            for feat in all_features:
                val_str = request.form.get(feat)
                if val_str:
                    val = float(val_str)
                    if val != 0:
                        fixed_features[feat] = val

            if target_CS < 10 or target_CS > 90:
                return render_template(
                    'optimize.html',
                    error=f"WARNING: Target CS {target_CS} MPa is outside reliable range (10–90 MPa).",
                    **template_vars
                )

            results = optimize_materials(target_CS, **fixed_features)

            if "Error" in results:
                return render_template('optimize.html',
                                       error=results["Error"],
                                       **template_vars)

            optimized_list = [
                (name_map.get(k, k), v)
                for k, v in results["Optimized Materials"].items()
            ]

            template_vars.update({
                'target_cs':     target_CS,
                'predicted_cs':  results["Predicted_CS"],
                'delta_cs':      results["Delta_CS"],
                'optimized_list': optimized_list,
            })

            return render_template('optimize.html',
                                   form_data=request.form,
                                   **template_vars)

        except ValueError:
            return render_template('optimize.html',
                                   error="Please enter valid numbers.",
                                   form_data=request.form,
                                   **template_vars)
        except Exception as e:
            app.logger.critical(f"Optimization error: {e}", exc_info=True)
            return render_template('optimize.html',
                                   error=f"Unexpected error: {str(e)}",
                                   form_data=request.form,
                                   **template_vars)

    return render_template('optimize.html',
                           form_data=request.form,
                           **template_vars)


@app.route('/chart', methods=['GET'])
def chart():
    logs = list_logs()
    return render_template('chart.html', logs=logs)


@app.route('/chart_data/<log_file>', methods=['GET'])
def chart_data(log_file):
    path = os.path.join(LOG_DIR, log_file)
    if not os.path.exists(path):
        return jsonify({'error': 'File not found'}), 404
    df   = pd.read_csv(path).rename(columns=name_map)
    return jsonify(df.to_dict(orient='records'))


@app.route('/upload_reality/<log_file>', methods=['POST'])
def upload_reality(log_file):
    file = request.files.get('reality_csv')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    path = os.path.join(LOG_DIR, log_file)
    try:
        df_pred   = pd.read_csv(path)
        df_actual = pd.read_csv(file)
    except FileNotFoundError:
        return jsonify({'error': 'Log file not found'}), 404
    except Exception as e:
        return jsonify({'error': f'Failed to read CSV: {e}'}), 400

    if len(df_pred) != len(df_actual):
        return jsonify({'error': 'Row count mismatch'}), 400
    if df_actual.shape[1] < 1:
        return jsonify({'error': 'Reality CSV must have at least 1 column'}), 400

    df_pred['y_true'] = df_actual.iloc[:, -1]
    df_pred.to_csv(path, index=False)
    app.logger.info(f"Reality data saved for {log_file}")
    return jsonify({'success': True})


@app.context_processor
def inject_optimize_status():
    return dict(optimize_enabled=OPTIMIZE_ENABLED)


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    app.logger.info(f"Starting server on port {PORT}...")
    app.run(host='0.0.0.0', port=PORT, debug=False, use_reloader=False)