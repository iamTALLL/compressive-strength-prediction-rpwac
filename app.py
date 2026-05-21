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
    OPTIMIZE_ENABLED = True
except ImportError:
    print("WARNING: mealpy not found. Inverse Design feature will be disabled.")
    OPTIMIZE_ENABLED = False

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

# FIX #1: Single source of truth for feature names.
# These SHORT names must match what the model was trained on.
# MODEL_FEATURES is removed; all routes now use `all_features`.
_FALLBACK_FEATURES = [
    "c_d", "ce_cs", "ce_sg", "f_q", "f_sg", "c_q", "c_sg",
    "w", "w/c", "p_q", "p_ts", "p_sg", "slump", "c_sg1"
]

_FULL_NAMES = [
    'Cement dosage', 'Cement compressive strength', 'Cement specific gravity',
    'Fine aggregate quantity', 'Fine aggregate specific gravity',
    'Coarse aggregate quantity', 'Coarse aggregate specific gravity',
    'Water', 'Water/cement', 'Plastic quantity',
    'Plastic tensile strength', 'Plastic specific gravity',
    'Slump', 'Concrete specific gravity'
]

# Keys used in fitness_func — must match all_features exactly (resolved after load)
# These are resolved dynamically after model load; defined here for IDE clarity.
KEY_CEMENT    = None  # resolved to all_features index for cement mass
KEY_CEMENT_SG = None
KEY_FINE_Q    = None
KEY_FINE_SG   = None
KEY_COARSE_Q  = None
KEY_COARSE_SG = None
KEY_PLASTIC_Q = None
KEY_PLASTIC_SG = None
KEY_WATER     = None

try:
    scaler = joblib.load('scaler.pkl')
    model  = joblib.load('best_inverse_design_xgboost.pkl')
    app.logger.info("Model and scaler loaded successfully.")

    try:
        all_features = model.feature_names_in_.tolist()
    except AttributeError:
        all_features = _FALLBACK_FEATURES[:]

    # Validate feature count matches scaler
    if hasattr(scaler, 'n_features_in_') and scaler.n_features_in_ != len(all_features):
        raise ValueError(
            f"Scaler expects {scaler.n_features_in_} features, "
            f"but model has {len(all_features)}."
        )

    name_map = dict(zip(all_features, _FULL_NAMES[:len(all_features)]))

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

    # FIX #3: Resolve feature keys dynamically from all_features so
    # fitness_func never hard-codes names that might not exist.
    # Map by position index (positional, not by name string).
    _FEAT_IDX = {f: i for i, f in enumerate(all_features)}

    # Attempt common naming conventions for each physical quantity.
    def _resolve_key(*candidates):
        for c in candidates:
            if c in _FEAT_IDX:
                return c
        return None  # feature not present in this model

    KEY_CEMENT     = _resolve_key('c_d',  'cement_dosage')
    KEY_CEMENT_SG  = _resolve_key('ce_sg','cement_specific_gravity')
    KEY_FINE_Q     = _resolve_key('f_q',  'fine_aggregate_quantity')
    KEY_FINE_SG    = _resolve_key('f_sg', 'fine_aggregate_specific_gravity')
    KEY_COARSE_Q   = _resolve_key('c_q',  'coarse_aggregate_quantity')
    KEY_COARSE_SG  = _resolve_key('c_sg1','c_sg', 'coarse_aggregate_specific_gravity')
    KEY_PLASTIC_Q  = _resolve_key('p_q',  'plastic_quantity')
    KEY_PLASTIC_SG = _resolve_key('p_sg', 'plastic_specific_gravity')
    KEY_WATER      = _resolve_key('w',    'water')

    app.logger.info("Feature bounds and key mapping created successfully.")

except FileNotFoundError as e:
    app.logger.critical(f"Model/scaler file not found: {e}")
    sys.exit(1)
except Exception as e:
    app.logger.critical(f"FATAL ERROR during initialization: {e}")
    sys.exit(1)


# ============================================================
# HELPER — use DataFrame to avoid feature-name warnings
# ============================================================
def predict_cs(full_vector):
    """Predict CS from full_vector (list of 14 values), returns float.
    FIX #5: always build DataFrame with all_features to guarantee correct
    column order matches what scaler/model expect.
    """
    df_input = pd.DataFrame([full_vector], columns=all_features)
    scaled   = scaler.transform(df_input)
    return float(model.predict(scaled)[0])


# ============================================================
# SECURITY HELPER
# ============================================================
def safe_log_path(log_file):
    """FIX #2: Prevent path traversal attacks.
    Returns resolved absolute path only if it stays inside LOG_DIR,
    otherwise returns None.
    """
    # Strip any directory components
    safe_name = os.path.basename(log_file)
    # Only allow .csv files
    if not safe_name.lower().endswith('.csv'):
        return None
    candidate = os.path.realpath(os.path.join(LOG_DIR, safe_name))
    log_dir_real = os.path.realpath(LOG_DIR)
    if not candidate.startswith(log_dir_real + os.sep) and candidate != log_dir_real:
        return None
    return candidate


# ============================================================
# OPTIMIZATION CONFIG — FIX #4: only defined when OPTIMIZE_ENABLED
# ============================================================
# Provide safe defaults so templates/routes never get NameError
L1 = L2 = None

if OPTIMIZE_ENABLED:
    PENALTY_FACTOR         = 5
    V_MIN_TARGET           = 0.90
    V_MAX_TARGET           = 1.10
    L1                     = 0.005
    L2                     = 0.0025
    MAX_RETRIES            = 2
    EPOCH_BASE             = 50
    EPOCH_RETRY            = 100
    POP_SIZE               = 20
    V_PLASTIC_ABSOLUTE_MAX = 0.20

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
            # Rebuild full 14-dim vector
            # FIX #5: build in the same order as all_features so
            # predict_cs() receives the correct positional vector.
            full_vector = [0.0] * len(all_features)
            for i, idx in enumerate(fixed_idx):
                full_vector[idx] = fixed_values[i]
            for i, idx in enumerate(free_idx):
                full_vector[idx] = x[i]

            # FIX #3: use resolved keys (may be None if feature absent)
            def _get(key, default=0.0):
                if key is None:
                    return default
                return full_vector[_FEAT_IDX[key]]

            c_d_mass = _get(KEY_CEMENT,    0.0)
            p_q_mass = _get(KEY_PLASTIC_Q, 0.0)
            f_q_mass = _get(KEY_FINE_Q,    0.0)
            c_q_mass = _get(KEY_COARSE_Q,  0.0)
            w_mass   = _get(KEY_WATER,     0.0)

            ce_sg = _get(KEY_CEMENT_SG,  1.0) or 1.0
            f_sg  = _get(KEY_FINE_SG,    1.0) or 1.0
            c_sg1 = _get(KEY_COARSE_SG,  1.0) or 1.0
            p_sg  = _get(KEY_PLASTIC_SG, 1.0) or 1.0

            V_cement     = c_d_mass / (ce_sg * 1000)
            V_fine_agg   = f_q_mass / (f_sg  * 1000)
            V_coarse_agg = c_q_mass / (c_sg1 * 1000)
            V_plastic    = p_q_mass / (p_sg  * 1000)
            V_water      = w_mass   / 1000
            V_total      = V_cement + V_fine_agg + V_coarse_agg + V_plastic + V_water

            pred_CS = predict_cs(full_vector)

            penalty_vol = 0.0
            if V_total < V_MIN_TARGET:
                penalty_vol = PENALTY_FACTOR * (V_MIN_TARGET - V_total)
            elif V_total > V_MAX_TARGET:
                penalty_vol = PENALTY_FACTOR * (V_total - V_MAX_TARGET)

            penalty_plastic = 0.0
            if V_plastic > V_PLASTIC_ABSOLUTE_MAX:
                penalty_plastic = PENALTY_FACTOR * 5 * (V_plastic - V_PLASTIC_ABSOLUTE_MAX)

            penalty_cs = 0.0
            if pred_CS < target_CS:
                penalty_cs = PENALTY_FACTOR * (target_CS - pred_CS)

            delta_CS = (pred_CS - target_CS) if pred_CS >= target_CS else 0.0

            fitness = (
                delta_CS
                + L1 * c_d_mass
                - L2 * p_q_mass
                + penalty_vol
                + penalty_plastic
                + penalty_cs
            )
            return fitness

        # ── Retry loop ──────────────────────────────────────────
        best_solution = None
        best_pred_CS  = -np.inf
        best_delta    = np.inf

        for attempt in range(1, MAX_RETRIES + 1):
            epoch = EPOCH_BASE if attempt == 1 else EPOCH_RETRY
            app.logger.info(
                f"Optimization attempt {attempt}/{MAX_RETRIES}, "
                f"epoch={epoch}, pop={POP_SIZE}"
            )

            problem_dict = {
                "obj_func": fitness_func,
                "bounds":   FloatVar(lb=lb, ub=ub),
                "minmax":   "min",
            }

            try:
                solver = OriginalFOX(epoch=epoch, pop_size=POP_SIZE)
                agent  = solver.solve(problem_dict)
                sol    = agent.solution

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

                if pred_CS >= target_CS and delta < best_delta:
                    best_delta    = delta
                    best_solution = full_vector[:]
                    best_pred_CS  = pred_CS
                    app.logger.info(f"  ✅ New best: pred={pred_CS:.2f}, delta={delta:.2f}")

                if pred_CS >= target_CS and delta < 2.0:
                    app.logger.info(f"  Early stop: delta={delta:.2f} < 2.0 MPa")
                    break

            except Exception as e:
                app.logger.error(f"Attempt {attempt} failed: {e}", exc_info=True)
                continue

        # Final heavy pass if no attempt reached target
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
                app.logger.info(
                    f"  Final pass: pred_CS={best_pred_CS:.2f}, delta={best_delta:.2f}"
                )

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
            # FIX #1: use all_features (same names as scaler/model) everywhere
            inputs = [float(request.form[feat]) for feat in all_features]

            if any(x < 0 for x in inputs):
                return render_template('predict.html',
                                       prediction=None,
                                       error="Please enter non-negative values.",
                                       features=all_features,
                                       name_map=name_map)

            df_input   = pd.DataFrame([inputs], columns=all_features)
            scaled     = scaler.transform(df_input)
            prediction = float(model.predict(scaled)[0])

            df_log = pd.DataFrame([inputs + [prediction]],
                                  columns=all_features + ['Predicted_c_cs'])
            timestamp    = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            log_filename = f'single_{timestamp}.csv'
            df_log.to_csv(os.path.join(LOG_DIR, log_filename), index=False)
            app.logger.info(f"Single prediction logged to {log_filename}")

            return render_template('predict.html',
                                   prediction=prediction, error=None,
                                   features=all_features,
                                   name_map=name_map)

        except KeyError as e:
            return render_template('predict.html',
                                   prediction=None,
                                   error=f"Missing field in form: {e}",
                                   features=all_features,
                                   name_map=name_map)
        except ValueError:
            return render_template('predict.html',
                                   prediction=None,
                                   error="Please enter valid numbers for all fields.",
                                   features=all_features,
                                   name_map=name_map)
        except Exception as e:
            app.logger.critical(f"Critical prediction error: {e}", exc_info=True)
            return render_template('predict.html',
                                   prediction=None,
                                   error=f"An unexpected error occurred: {str(e)}",
                                   features=all_features,
                                   name_map=name_map)

    return render_template('predict.html',
                           prediction=None, error=None,
                           features=all_features,
                           name_map=name_map)


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

        # FIX #6: validate file extension
        if not file.filename.lower().endswith('.csv'):
            return render_template('predict_csv.html',
                                   predictions=None,
                                   error="Only .csv files are accepted.")

        try:
            df = pd.read_csv(file)

            if df.shape[1] != len(all_features):
                error = (f"CSV must have {len(all_features)} columns, "
                         f"found {df.shape[1]}. "
                         f"Expected: {', '.join(all_features)}")
                return render_template('predict_csv.html',
                                       predictions=None, error=error)

            df.columns = all_features

            # FIX #8: warn when NaN values are present instead of silently filling
            nan_count = df.isnull().sum().sum()
            if nan_count > 0:
                app.logger.warning(
                    f"Batch CSV contains {nan_count} missing value(s). "
                    "Filling with 0 — check your data."
                )
            df = df.fillna(0)

            scaled_X             = scaler.transform(df)
            predictions          = model.predict(scaled_X)
            df['Predicted_c_cs'] = predictions

            timestamp    = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            log_filename = f'batch_{timestamp}.csv'
            df.to_csv(os.path.join(LOG_DIR, log_filename), index=False)
            app.logger.info(f"Batch prediction logged to {log_filename}")

            display_df = df.rename(columns=name_map).round(4)
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
    # FIX #4: L1/L2 are None when OPTIMIZE_ENABLED is False — pass safely
    template_vars = {
        'all_features':     all_features,
        'name_map':         name_map,
        'optimize_enabled': OPTIMIZE_ENABLED,
        'L1':               L1,
        'L2':               L2,
    }

    if not OPTIMIZE_ENABLED:
        return render_template('optimize.html',
                               error="Inverse Design is disabled (mealpy not found).",
                               **template_vars)

    if request.method == 'POST':
        app.logger.info("Received POST for Inverse Design.")
        try:
            target_CS      = float(request.form.get('target_cs'))
            fixed_features = {}

            for feat in all_features:
                val_str = request.form.get(feat)
                if val_str:
                    try:
                        val = float(val_str)
                    except ValueError:
                        return render_template('optimize.html',
                                               error=f"Invalid value for field '{feat}'.",
                                               form_data=request.form,
                                               **template_vars)
                    if val != 0:
                        fixed_features[feat] = val

            if target_CS < 10 or target_CS > 90:
                return render_template(
                    'optimize.html',
                    error=(f"WARNING: Target CS {target_CS} MPa is outside "
                           "reliable range (10–90 MPa)."),
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
                'target_cs':      target_CS,
                'predicted_cs':   results["Predicted_CS"],
                'delta_cs':       results["Delta_CS"],
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
    # FIX #2: sanitize path to prevent traversal attacks
    path = safe_log_path(log_file)
    if path is None:
        return jsonify({'error': 'Invalid or disallowed filename'}), 400
    if not os.path.exists(path):
        return jsonify({'error': 'File not found'}), 404
    df = pd.read_csv(path).rename(columns=name_map)
    return jsonify(df.to_dict(orient='records'))


@app.route('/upload_reality/<log_file>', methods=['POST'])
def upload_reality(log_file):
    # FIX #2: sanitize path
    path = safe_log_path(log_file)
    if path is None:
        return jsonify({'error': 'Invalid or disallowed filename'}), 400

    file = request.files.get('reality_csv')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    # FIX #6: validate uploaded file extension
    if not file.filename.lower().endswith('.csv'):
        return jsonify({'error': 'Only .csv files are accepted'}), 400

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

    df_pred['y_true'] = df_actual.iloc[:, -1].values
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
    # FIX #7: use_reloader=False is intentional when debug=False (avoids double-init).
    # If debug=True is ever enabled, remove use_reloader=False.
    app.run(host='0.0.0.0', port=PORT, debug=False, use_reloader=False)