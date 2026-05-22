from flask import Flask, render_template, request, jsonify
import os
import sys
import joblib
import logging
import warnings
import datetime
import threading
import time

from logging.handlers import RotatingFileHandler

# Keep numerical libraries modest on Render free instances.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# =========================================================
# OPTIONAL OPTIMIZER
# =========================================================
try:
    from mealpy.swarm_based.FOX import OriginalFOX
    from mealpy.utils.space import FloatVar

    MEALPY_ENABLED = True

except ImportError:
    print("WARNING: mealpy not installed.")
    MEALPY_ENABLED = False

OPTIMIZE_ENABLED = MEALPY_ENABLED

# =========================================================
# CONFIG
# =========================================================
PORT = 5004
LOG_DIR = "logs"

os.makedirs(LOG_DIR, exist_ok=True)

# =========================================================
# LOGGER
# =========================================================
handler = RotatingFileHandler(
    os.path.join(LOG_DIR, "app_system.log"),
    maxBytes=100_000,
    backupCount=3
)

handler.setLevel(logging.INFO)

handler.setFormatter(
    logging.Formatter(
        "%(asctime)s - [%(levelname)s] - %(message)s"
    )
)

app = Flask(__name__, template_folder="templates")

app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)

logging.getLogger("mealpy").setLevel(logging.INFO)
logging.getLogger("mealpy.swarm_based.FOX").setLevel(logging.INFO)

# =========================================================
# LOAD MODEL + SCALER + BOUNDS
# =========================================================
try:
    model = joblib.load("best_inverse_design_xgboost.pkl")
    scaler = joblib.load("feature_scaler.pkl")
    feature_bounds = joblib.load("feature_bounds.pkl")

    app.logger.info("Model loaded successfully.")

except Exception as e:
    app.logger.critical(f"Cannot load model files: {e}")
    sys.exit(1)

# =========================================================
# FEATURES
# =========================================================
try:
    all_features = scaler.feature_names_in_.tolist()
except AttributeError:
    try:
        all_features = model.feature_names_in_.tolist()
    except AttributeError:
        all_features = list(feature_bounds.keys())

all_features = [str(f) for f in all_features]

_FULL_NAMES = [
    "Cement dosage",
    "Cement compressive strength",
    "Cement specific gravity",
    "Fine aggregate quantity",
    "Fine aggregate specific gravity",
    "Coarse aggregate quantity",
    "Coarse aggregate specific gravity",
    "Water",
    "Water/cement",
    "Plastic quantity",
    "Plastic tensile strength",
    "Plastic specific gravity",
    "Slump",
    "Concrete specific gravity"
]

name_map = dict(
    zip(
        all_features,
        _FULL_NAMES[:len(all_features)]
    )
)

_FEAT_IDX = {
    f: i for i, f in enumerate(all_features)
}

# =========================================================
# KEY RESOLVE
# =========================================================
def _resolve_key(*candidates):
    for c in candidates:
        if c in _FEAT_IDX:
            return c

    return None


KEY_CEMENT = _resolve_key("c_d")
KEY_CEMENT_CS = _resolve_key("ce_cs")
KEY_CEMENT_SG = _resolve_key("ce_sg")

KEY_FINE_Q = _resolve_key("f_q")
KEY_FINE_SG = _resolve_key("f_sg")

KEY_COARSE_Q = _resolve_key("c_q")
KEY_COARSE_SG = _resolve_key("c_sg", "c_sg1")

KEY_WATER = _resolve_key("w")
KEY_WC = _resolve_key("w/c", "w_c")

KEY_PLASTIC_Q = _resolve_key("p_q")
KEY_PLASTIC_TS = _resolve_key("p_ts")
KEY_PLASTIC_SG = _resolve_key("p_sg")

KEY_SLUMP = _resolve_key("slump")
KEY_CONCRETE_SG = _resolve_key("c_sg1")

# =========================================================
# REALISTIC DEFAULTS + LIMITS
# =========================================================
AUTO_FIXED_VALUES = {
    "ce_sg": 3.05,
    "f_sg": 2.795,
    "c_sg": 2.7645,
    "p_sg": 1.495,
    "p_ts": 410.5,
    "slump": 12.5
}

REALISTIC_LIMITS = {
    "c_d": (280.0, 500.0),
    "ce_cs": (25.0, 60.0),
    "f_q": (500.0, 950.0),
    "c_q": (850.0, 1250.0),
    "w": (130.0, 200.0),
    "p_q": (0.0, 40.0),
    "c_sg1": (2.20, 2.55)
}

WC_MIN = 0.30
WC_MAX = 0.50
VOLUME_MIN = 0.99
VOLUME_MAX = 1.01
PLASTIC_VOLUME_MAX = 0.030
REALISTIC_TARGET_MAX = 50.0
FINE_COARSE_RATIO_MIN = 0.45
FINE_COARSE_RATIO_MAX = 0.95

# =========================================================
# OPTIMIZATION CONFIG
# =========================================================
USE_MEALPY_OPTIMIZER = os.getenv("USE_MEALPY_OPTIMIZER", "1") == "1"
MAX_RETRIES = int(os.getenv("OPT_MAX_RETRIES", "3"))
EPOCH_BASE = int(os.getenv("OPT_EPOCH_BASE", "250"))
EPOCH_RETRY = int(os.getenv("OPT_EPOCH_RETRY", "600"))
POP_SIZE = int(os.getenv("OPT_POP_SIZE", "60"))
OPT_RANDOM_SAMPLES = int(os.getenv("OPT_RANDOM_SAMPLES", "800"))
OPT_LOCAL_SAMPLES = int(os.getenv("OPT_LOCAL_SAMPLES", "250"))
OPTIMIZE_TIMEOUT_SECONDS = int(os.getenv("OPT_TIMEOUT_SECONDS", "600"))
OPTIMIZER_LOCK = threading.Lock()

# =========================================================
# HELPERS
# =========================================================
def _get(full_vector, key, default=0.0):
    if key is None:
        return default

    return float(full_vector[_FEAT_IDX[key]])


def _set(full_vector, key, value):
    if key is not None:
        full_vector[_FEAT_IDX[key]] = float(value)


def compute_wc(cement, water):
    return float(water) / max(float(cement), 1e-6)


def compute_concrete_sg(full_vector):
    mass = (
        _get(full_vector, KEY_CEMENT, 0.0)
        + _get(full_vector, KEY_FINE_Q, 0.0)
        + _get(full_vector, KEY_COARSE_Q, 0.0)
        + _get(full_vector, KEY_WATER, 0.0)
        + _get(full_vector, KEY_PLASTIC_Q, 0.0)
    )

    return mass / 1000.0


def force_physics(full_vector):
    cement = _get(full_vector, KEY_CEMENT, 0.0)
    water = _get(full_vector, KEY_WATER, 0.0)

    if KEY_WC is not None:
        _set(full_vector, KEY_WC, compute_wc(cement, water))

    if (
        KEY_CONCRETE_SG is not None
        and KEY_CONCRETE_SG != KEY_COARSE_SG
    ):
        _set(full_vector, KEY_CONCRETE_SG, compute_concrete_sg(full_vector))

    return full_vector


def compute_volume(full_vector):
    cement = _get(full_vector, KEY_CEMENT, 0.0)
    water = _get(full_vector, KEY_WATER, 0.0)
    fine = _get(full_vector, KEY_FINE_Q, 0.0)
    coarse = _get(full_vector, KEY_COARSE_Q, 0.0)
    plastic = _get(full_vector, KEY_PLASTIC_Q, 0.0)

    ce_sg = max(_get(full_vector, KEY_CEMENT_SG, 3.05), 0.1)
    f_sg = max(_get(full_vector, KEY_FINE_SG, 2.795), 0.1)
    c_sg = max(_get(full_vector, KEY_COARSE_SG, 2.7645), 0.1)
    p_sg = max(_get(full_vector, KEY_PLASTIC_SG, 1.495), 0.1)

    v_cement = cement / (ce_sg * 1000.0)
    v_water = water / 1000.0
    v_fine = fine / (f_sg * 1000.0)
    v_coarse = coarse / (c_sg * 1000.0)
    v_plastic = plastic / (p_sg * 1000.0)

    return {
        "cement": v_cement,
        "water": v_water,
        "fine": v_fine,
        "coarse": v_coarse,
        "plastic": v_plastic,
        "total": v_cement + v_water + v_fine + v_coarse + v_plastic
    }


def predict_cs(full_vector):
    full_vector = force_physics(list(full_vector))

    arr = np.asarray([full_vector], dtype=float)
    scaled = scaler.transform(arr)
    pred = model.predict(scaled)[0]

    return float(pred)


def safe_log_path(log_file):
    safe_name = os.path.basename(log_file)

    if not safe_name.lower().endswith(".csv"):
        return None

    candidate = os.path.realpath(
        os.path.join(LOG_DIR, safe_name)
    )

    log_dir_real = os.path.realpath(LOG_DIR)

    if (
        not candidate.startswith(log_dir_real + os.sep)
        and candidate != log_dir_real
    ):
        return None

    return candidate


def list_logs():
    return sorted(
        f for f in os.listdir(LOG_DIR)
        if f.endswith(".csv")
    )


def _bounded_feature_range(feat):
    lo = float(feature_bounds[feat]["min"])
    hi = float(feature_bounds[feat]["max"])

    if feat in REALISTIC_LIMITS:
        rlo, rhi = REALISTIC_LIMITS[feat]
        lo = max(lo, rlo)
        hi = min(hi, rhi)

    if lo > hi:
        lo = float(feature_bounds[feat]["min"])
        hi = float(feature_bounds[feat]["max"])

    return lo, hi


def _default_fixed_value(feat):
    if feat in AUTO_FIXED_VALUES:
        return AUTO_FIXED_VALUES[feat]

    lo = float(feature_bounds[feat]["min"])
    hi = float(feature_bounds[feat]["max"])

    return (lo + hi) / 2.0


def _derived_feature_keys():
    keys = set()

    if KEY_WC is not None:
        keys.add(KEY_WC)

    if (
        KEY_CONCRETE_SG is not None
        and KEY_CONCRETE_SG != KEY_COARSE_SG
    ):
        keys.add(KEY_CONCRETE_SG)

    return keys


def evaluate_practicality(full_vector, target_CS=None):
    full_vector = force_physics(list(full_vector))
    volume = compute_volume(full_vector)

    cement = _get(full_vector, KEY_CEMENT, 0.0)
    water = _get(full_vector, KEY_WATER, 0.0)
    fine = _get(full_vector, KEY_FINE_Q, 0.0)
    coarse = _get(full_vector, KEY_COARSE_Q, 0.0)
    plastic = _get(full_vector, KEY_PLASTIC_Q, 0.0)
    wc_ratio = compute_wc(cement, water)
    fine_coarse_ratio = fine / max(coarse, 1e-6)

    warnings_list = []
    failed = False

    if wc_ratio < WC_MIN or wc_ratio > WC_MAX:
        failed = True
        warnings_list.append(
            f"w/c = {wc_ratio:.3f} is outside {WC_MIN:.2f}-{WC_MAX:.2f}."
        )
    elif target_CS is not None and target_CS >= 40 and wc_ratio > 0.48:
        warnings_list.append(
            "w/c is high for concrete near or above 40 MPa."
        )

    if volume["total"] < VOLUME_MIN or volume["total"] > VOLUME_MAX:
        warnings_list.append(
            f"Total absolute volume = {volume['total']:.3f} m3, not close to 1.000 m3."
        )

        if volume["total"] < 0.97 or volume["total"] > 1.03:
            failed = True

    if water > 200:
        failed = True
        warnings_list.append("Water content is above 200 kg/m3.")

    if cement > 500:
        failed = True
        warnings_list.append("Cement dosage is above 500 kg/m3.")
    elif cement > 450:
        warnings_list.append("Cement dosage is high; check heat, shrinkage, and cost.")

    if volume["plastic"] > PLASTIC_VOLUME_MAX:
        warnings_list.append(
            f"Plastic volume = {volume['plastic']:.3f} m3 is above the preferred limit."
        )

        if volume["plastic"] > 0.05:
            failed = True

    if plastic > 40:
        failed = True
        warnings_list.append("Plastic quantity is above 40 kg/m3.")

    if (
        fine_coarse_ratio < FINE_COARSE_RATIO_MIN
        or fine_coarse_ratio > FINE_COARSE_RATIO_MAX
    ):
        warnings_list.append(
            f"Fine/coarse ratio = {fine_coarse_ratio:.3f}, outside the preferred range."
        )

    if failed:
        status = "NOT PRACTICAL"
    elif warnings_list:
        status = "WARNING"
    else:
        status = "PASS"

    return {
        "status": status,
        "warnings": warnings_list,
        "wc_ratio": wc_ratio,
        "total_volume": volume["total"],
        "plastic_volume": volume["plastic"],
        "concrete_sg": compute_concrete_sg(full_vector),
        "fine_coarse_ratio": fine_coarse_ratio
    }

# =========================================================
# OPTIMIZER
# =========================================================
if OPTIMIZE_ENABLED:

    def optimize_materials(target_CS, **fixed_features):
        if not OPTIMIZER_LOCK.acquire(blocking=False):
            return {
                "Error": (
                    "Another optimization is still running. Please wait "
                    "for it to finish before submitting again."
                ),
                "Predicted_CS": None
            }

        started_at = time.monotonic()

        try:
            return _optimize_materials_locked(
                target_CS,
                started_at,
                **fixed_features
            )

        finally:
            OPTIMIZER_LOCK.release()


    def _optimize_materials_locked(target_CS, started_at, **fixed_features):
        if target_CS > REALISTIC_TARGET_MAX:
            return {
                "Error": (
                    f"Target {target_CS} MPa exceeds realistic model "
                    f"range ({REALISTIC_TARGET_MAX} MPa)."
                ),
                "Predicted_CS": None
            }

        derived_keys = _derived_feature_keys()

        for key in derived_keys:
            fixed_features.pop(key, None)

        for feat, value in AUTO_FIXED_VALUES.items():
            if feat in feature_bounds and feat not in fixed_features:
                fixed_features[feat] = value

        fixed_features = {
            k: v for k, v in fixed_features.items()
            if k in _FEAT_IDX
        }

        fixed_idx = [
            _FEAT_IDX[k]
            for k in fixed_features.keys()
        ]

        fixed_values = [
            float(v)
            for v in fixed_features.values()
        ]

        free_idx = [
            i for i in range(len(all_features))
            if (
                i not in fixed_idx
                and all_features[i] in feature_bounds
                and all_features[i] not in derived_keys
            )
        ]

        if not free_idx:
            return {
                "Error": "All variables are fixed.",
                "Predicted_CS": None
            }

        lb = []
        ub = []

        for i in free_idx:
            lo, hi = _bounded_feature_range(all_features[i])
            lb.append(lo)
            ub.append(hi)

        def build_vector(x):
            full_vector = [
                _default_fixed_value(feat)
                if feat in feature_bounds
                else 0.0
                for feat in all_features
            ]

            for i, idx in enumerate(fixed_idx):
                full_vector[idx] = fixed_values[i]

            for i, idx in enumerate(free_idx):
                full_vector[idx] = x[i]

            return force_physics(full_vector)

        def score_mix(full_vector, pred_CS):
            cement = _get(full_vector, KEY_CEMENT, 0.0)
            water = _get(full_vector, KEY_WATER, 0.0)
            plastic = _get(full_vector, KEY_PLASTIC_Q, 0.0)
            fine = _get(full_vector, KEY_FINE_Q, 0.0)
            coarse = _get(full_vector, KEY_COARSE_Q, 0.0)
            wc_ratio = compute_wc(cement, water)

            volume = compute_volume(full_vector)
            total_volume = volume["total"]
            plastic_volume = volume["plastic"]

            gap = max(0.0, target_CS - pred_CS)
            overshoot = max(0.0, pred_CS - target_CS)

            penalty_target = (gap ** 2) * 180.0
            penalty_over = overshoot * 6.0

            penalty_wc = 0.0
            if wc_ratio < WC_MIN:
                penalty_wc += ((WC_MIN - wc_ratio) ** 2) * 120000.0
            if wc_ratio > WC_MAX:
                penalty_wc += ((wc_ratio - WC_MAX) ** 2) * 120000.0

            if target_CS >= 40.0 and wc_ratio > 0.48:
                penalty_wc += ((wc_ratio - 0.48) ** 2) * 60000.0

            penalty_volume = 0.0
            if total_volume < VOLUME_MIN:
                penalty_volume += ((VOLUME_MIN - total_volume) ** 2) * 250000.0
            if total_volume > VOLUME_MAX:
                penalty_volume += ((total_volume - VOLUME_MAX) ** 2) * 250000.0

            penalty_plastic = 0.0
            if plastic_volume > PLASTIC_VOLUME_MAX:
                penalty_plastic += (
                    (plastic_volume - PLASTIC_VOLUME_MAX) ** 2
                ) * 250000.0

            penalty_material_balance = 0.0
            if fine < FINE_COARSE_RATIO_MIN * coarse:
                penalty_material_balance += (
                    FINE_COARSE_RATIO_MIN * coarse - fine
                ) * 3.0
            if fine > FINE_COARSE_RATIO_MAX * coarse:
                penalty_material_balance += (
                    fine - FINE_COARSE_RATIO_MAX * coarse
                ) * 3.0

            cement_cost = cement * 0.01
            water_cost = water * 0.004
            plastic_cost = plastic * 0.03

            return (
                penalty_target
                + penalty_over
                + penalty_wc
                + penalty_volume
                + penalty_plastic
                + penalty_material_balance
                + cement_cost
                + water_cost
                + plastic_cost
            )

        def fitness_func(x):
            full_vector = build_vector(x)
            pred_CS = predict_cs(full_vector)

            return score_mix(full_vector, pred_CS)

        def predict_many(vectors):
            arr = np.asarray(vectors, dtype=float)
            scaled = scaler.transform(arr)

            return model.predict(scaled)

        def fast_random_solver():
            lb_arr = np.asarray(lb, dtype=float)
            ub_arr = np.asarray(ub, dtype=float)
            span = ub_arr - lb_arr
            rng = np.random.default_rng()

            best_solution = None
            best_pred = None
            best_score = None

            def evaluate_candidates(xs):
                nonlocal best_solution, best_pred, best_score

                vectors = [
                    build_vector(x)
                    for x in xs
                ]
                preds = predict_many(vectors)

                for vector, pred in zip(vectors, preds):
                    score = score_mix(vector, float(pred))

                    if best_score is None or score < best_score:
                        best_score = score
                        best_pred = float(pred)
                        best_solution = vector[:]

            fixed_seed = np.clip(
                (lb_arr + ub_arr) / 2.0,
                lb_arr,
                ub_arr
            )
            evaluate_candidates(np.asarray([fixed_seed]))

            random_xs = rng.uniform(
                lb_arr,
                ub_arr,
                size=(OPT_RANDOM_SAMPLES, len(free_idx))
            )
            evaluate_candidates(random_xs)

            if best_solution is not None and OPT_LOCAL_SAMPLES > 0:
                center = np.asarray(
                    [
                        best_solution[idx]
                        for idx in free_idx
                    ],
                    dtype=float
                )
                local_scale = span * 0.12
                local_xs = center + rng.normal(
                    0.0,
                    local_scale,
                    size=(OPT_LOCAL_SAMPLES, len(free_idx))
                )
                local_xs = np.clip(local_xs, lb_arr, ub_arr)
                evaluate_candidates(local_xs)

            if best_solution is None:
                raise RuntimeError("Optimizer could not generate a solution.")

            return best_solution, best_pred, best_score

        def run_mealpy_solver(epoch, pop_size):
            if not MEALPY_ENABLED:
                raise RuntimeError("mealpy is not installed.")

            problem_dict = {
                "obj_func": fitness_func,
                "bounds": FloatVar(
                    lb=lb,
                    ub=ub
                ),
                "minmax": "min",
                "log_to": None
            }

            solver = OriginalFOX(
                epoch=epoch,
                pop_size=pop_size
            )

            agent = solver.solve(problem_dict)
            full_vector = build_vector(agent.solution)
            pred = predict_cs(full_vector)

            return full_vector, pred, fitness_func(agent.solution)

        best_solution = None
        best_pred = None
        best_score = None

        if not USE_MEALPY_OPTIMIZER:
            try:
                best_solution, best_pred, best_score = fast_random_solver()
                app.logger.info(
                    f"Fast optimization predicted CS = {best_pred:.2f}, "
                    f"score = {best_score:.4f}"
                )

            except Exception as e:
                app.logger.error(f"Optimization failed: {e}")

        else:
            for attempt in range(1, MAX_RETRIES + 1):
                if time.monotonic() - started_at > OPTIMIZE_TIMEOUT_SECONDS:
                    break

                epoch = EPOCH_BASE if attempt == 1 else EPOCH_RETRY

                try:
                    app.logger.info(f"Optimization attempt {attempt}")

                    sol, pred, score = run_mealpy_solver(epoch, POP_SIZE)

                    app.logger.info(
                        f"Predicted CS = {pred:.2f}, score = {score:.4f}"
                    )

                    if best_score is None or score < best_score:
                        best_score = score
                        best_pred = pred
                        best_solution = sol[:]

                    if abs(pred - target_CS) < 0.8:
                        break

                except Exception as e:
                    app.logger.error(f"Optimization failed: {e}")

        if best_solution is None:
            try:
                best_solution, best_pred, best_score = fast_random_solver()

            except Exception as e:
                return {
                    "Error": str(e),
                    "Predicted_CS": None
                }

        volume = compute_volume(best_solution)
        wc_ratio = compute_wc(
            _get(best_solution, KEY_CEMENT, 0.0),
            _get(best_solution, KEY_WATER, 0.0)
        )
        practical = evaluate_practicality(best_solution, target_CS)

        return {
            "Optimized Materials": {
                feat: round(val, 4)
                for feat, val in zip(all_features, best_solution)
            },
            "Predicted_CS": round(best_pred, 2),
            "Delta_CS": round(best_pred - target_CS, 2),
            "Water_Cement_Ratio": round(wc_ratio, 4),
            "Total_Volume": round(volume["total"], 4),
            "Plastic_Volume": round(volume["plastic"], 4),
            "Concrete_SG": round(practical["concrete_sg"], 4),
            "Fine_Coarse_Ratio": round(practical["fine_coarse_ratio"], 4),
            "Practical_Status": practical["status"],
            "Practical_Warnings": practical["warnings"]
        }

# =========================================================
# ROUTES
# =========================================================
@app.route("/")
@app.route("/home")
def home():
    return render_template(
        "home.html",
        optimize_enabled=OPTIMIZE_ENABLED
    )

# =========================================================
# SINGLE PREDICTION
# =========================================================
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            inputs = [
                float(request.form[feat])
                for feat in all_features
            ]

            inputs = force_physics(inputs)
            prediction = predict_cs(inputs)

            df_log = pd.DataFrame(
                [inputs + [prediction]],
                columns=all_features + ["Predicted_c_cs"]
            )

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"single_{timestamp}.csv"

            df_log.to_csv(
                os.path.join(LOG_DIR, log_filename),
                index=False
            )

            return render_template(
                "predict.html",
                prediction=prediction,
                error=None,
                features=all_features,
                name_map=name_map
            )

        except Exception as e:
            return render_template(
                "predict.html",
                prediction=None,
                error=str(e),
                features=all_features,
                name_map=name_map
            )

    return render_template(
        "predict.html",
        prediction=None,
        error=None,
        features=all_features,
        name_map=name_map
    )

# =========================================================
# BATCH PREDICTION
# =========================================================
@app.route("/predict_csv", methods=["GET", "POST"])
def predict_csv():
    if request.method == "POST":
        try:
            file = request.files["file"]
            df = pd.read_csv(file)

            df.columns = all_features
            df = df.astype(float)

            if KEY_WC is not None and KEY_CEMENT is not None and KEY_WATER is not None:
                df[KEY_WC] = df[KEY_WATER] / df[KEY_CEMENT].clip(lower=1e-6)

            if (
                KEY_CONCRETE_SG is not None
                and KEY_CONCRETE_SG != KEY_COARSE_SG
            ):
                mass_cols = [
                    key for key in [
                        KEY_CEMENT,
                        KEY_FINE_Q,
                        KEY_COARSE_Q,
                        KEY_WATER,
                        KEY_PLASTIC_Q
                    ]
                    if key is not None
                ]
                df[KEY_CONCRETE_SG] = df[mass_cols].sum(axis=1) / 1000.0

            scaled = scaler.transform(df)
            preds = model.predict(scaled)

            df["Predicted_c_cs"] = preds

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"batch_{timestamp}.csv"

            df.to_csv(
                os.path.join(LOG_DIR, log_filename),
                index=False
            )

            display_df = df.rename(columns=name_map).round(4)

            return render_template(
                "predict_csv.html",
                predictions=display_df.to_html(index=False),
                error=None
            )

        except Exception as e:
            return render_template(
                "predict_csv.html",
                predictions=None,
                error=str(e)
            )

    return render_template(
        "predict_csv.html",
        predictions=None,
        error=None
    )

# =========================================================
# INVERSE DESIGN
# =========================================================
@app.route("/inverse-design", methods=["GET", "POST"])
@app.route("/optimize", methods=["GET", "POST"])
def inverse_design():
    template_vars = {
        "all_features": all_features,
        "name_map": name_map,
        "optimize_enabled": OPTIMIZE_ENABLED,
        "derived_features": _derived_feature_keys(),
        "WC_MIN": WC_MIN,
        "WC_MAX": WC_MAX,
        "VOLUME_MIN": VOLUME_MIN,
        "VOLUME_MAX": VOLUME_MAX
    }

    if not OPTIMIZE_ENABLED:
        return render_template(
            "optimize.html",
            error="mealpy not installed.",
            form_data={},
            **template_vars
        )

    if request.method == "POST":
        try:
            target_CS = float(request.form.get("target_cs"))
            fixed_features = {}
            derived_keys = _derived_feature_keys()

            for feat in all_features:
                if feat in derived_keys:
                    continue

                val_str = request.form.get(feat, "").strip()

                if val_str:
                    fixed_features[feat] = float(val_str)

            results = optimize_materials(
                target_CS,
                **fixed_features
            )

            if "Error" in results:
                return render_template(
                    "optimize.html",
                    error=results["Error"],
                    form_data=request.form,
                    **template_vars
                )

            optimized_list = [
                (
                    name_map.get(k, k),
                    v
                )
                for k, v in results["Optimized Materials"].items()
            ]

            template_vars.update({
                "target_cs": target_CS,
                "predicted_cs": results["Predicted_CS"],
                "delta_cs": results["Delta_CS"],
                "optimized_list": optimized_list,
                "water_cement_ratio": results["Water_Cement_Ratio"],
                "total_volume": results["Total_Volume"],
                "plastic_volume": results["Plastic_Volume"],
                "concrete_sg": results["Concrete_SG"],
                "fine_coarse_ratio": results["Fine_Coarse_Ratio"],
                "practical_status": results["Practical_Status"],
                "practical_warnings": results["Practical_Warnings"]
            })

            return render_template(
                "optimize.html",
                form_data=request.form,
                **template_vars
            )

        except Exception as e:
            return render_template(
                "optimize.html",
                error=str(e),
                form_data=request.form,
                **template_vars
            )

    return render_template(
        "optimize.html",
        form_data={},
        **template_vars
    )

# =========================================================
# CHART
# =========================================================
@app.route("/chart")
def chart():
    logs = list_logs()

    return render_template(
        "chart.html",
        logs=logs,
        optimize_enabled=OPTIMIZE_ENABLED
    )

# =========================================================
# CHART DATA
# =========================================================
@app.route("/chart_data/<log_file>")
def chart_data(log_file):
    path = safe_log_path(log_file)

    if path is None:
        return jsonify({
            "error": "Invalid filename"
        }), 400

    if not os.path.exists(path):
        return jsonify({
            "error": "File not found"
        }), 404

    df = pd.read_csv(path)

    return jsonify(
        df.to_dict(orient="records")
    )

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    app.logger.info(
        f"Starting server on port {PORT}"
    )

    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=False,
        use_reloader=False
    )
