import numpy as np
import pandas as pd
import joblib
import gradio as gr
from mealpy.evolutionary_based.GA import BaseGA
from mealpy import FOX
from mealpy.utils.problem import Problem
from mealpy.utils.space import FloatVar

# ==== 1. Load scaler + model ====
try:
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("xgb_woa_best_model.pkl")
except FileNotFoundError as e:
    print(f"Lỗi: Không tìm thấy file mô hình/scaler: {e}")
    # Thoát nếu không tìm thấy, vì không thể tiếp tục mà không có mô hình
    exit()

# ==== 2. Feature list ====
try:
    all_features = model.feature_names_in_.tolist()
except AttributeError:
    # Danh sách 14 features 
    all_features = [
        "c_d", "ce_cs", "ce_sg", "f_q", "f_sg", "c_q", "ca_sg", 
        "w", "w/c", "p_q", "p_ts", "p_sg", "slump", "c_sg"
    ]

name = [
    'Cement dosage',
    'Cement specific gravity',
    'Cement compressive strength',
    'Fine aggregate quantity',
    'Fine aggregate specific gravity',
    'Coarse aggregate quantity',
    'Corse aggregate specific gravity',
    'Water',
    'Water/cement',
    'Plastic quantity',
    'Plastic tensile strength',
    'Plastic specific gravity',
    'Slump',
    'Concrete specific gravity',
    'Concrete compressive strength'

    ]

name_map = dict(zip(all_features, name))

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

print("Feature bounds created.")

# ==== 4. Hàm tối ưu nguyên liệu bằng GA, các thông số cho ra khối bê tông có thể tích 1m3, thể tích xi măng nhỏ nhất và thể tích nhựa lớn nhất (nhưng không quá 20%) ====
PENALTY_FACTOR = 50000 
V_PLASTIC_ABSOLUTE_MAX = 0.20
V_MIN_TARGET = 0.95
V_MAX_TARGET = 1.05
L1 = 0.005
L2 = 0.0025

def optimize_materials(target_CS, **fixed_features):
    fixed_idx = [all_features.index(k) for k in fixed_features.keys()]
    fixed_values = [float(v) for v in fixed_features.values()]
    free_idx = [i for i in range(len(all_features)) if i not in fixed_idx]

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
            full_vector = [0]*len(all_features)
            for i, idx in enumerate(fixed_idx):
                full_vector[idx] = fixed_values[i]
            for i, idx in enumerate(free_idx):
                full_vector[idx] = candidate[i]
            
            feature_map = dict(zip(all_features, full_vector))
            c_d_mass = feature_map.get('c_d', 0)
            p_q_mass = feature_map.get('p_q', 0)
            f_q_mass = feature_map.get('f_q', 0)
            c_q_mass = feature_map.get('c_q', 0)
            w_mass = feature_map.get('w', 0)
            
            # Lấy và XỬ LÝ các giá trị Tỷ trọng (tránh chia cho 0)
            # Nếu tỷ trọng là 0, đặt về 1.0 (hoặc giá trị min/default hợp lý) để tránh lỗi.
            ce_sg = feature_map.get('ce_sg', 1.0)
            f_sg = feature_map.get('f_sg', 1.0)
            ca_sg = feature_map.get('ca_sg', 1.0)
            p_sg = feature_map.get('p_sg', 1.0)

            # Đảm bảo không có mẫu số nào bằng 0
            ce_sg = ce_sg if ce_sg != 0 else 1.0
            f_sg = f_sg if f_sg != 0 else 1.0
            ca_sg = ca_sg if ca_sg != 0 else 1.0
            p_sg = p_sg if p_sg != 0 else 1.0
            
            # Tính thể tích từng thành phần (V = M / (SG * 1000))
            V_cement = c_d_mass / (ce_sg * 1000)
            V_fine_agg = f_q_mass / (f_sg * 1000)
            V_coarse_agg = c_q_mass / (ca_sg * 1000)
            V_plastic = p_q_mass / (p_sg * 1000)
            V_water = w_mass / 1000 # RHO_WATER = 1000

            V_total = V_cement + V_fine_agg + V_coarse_agg + V_plastic + V_water

            # --- TÍNH HÌNH PHẠT (Penalty Logic) ---
            penalty_total_volume = 0
            if V_total < V_MIN_TARGET:
                violation = V_MIN_TARGET - V_total
                penalty_total_volume = PENALTY_FACTOR * violation
            elif V_total > V_MAX_TARGET:
                violation = V_total - V_MAX_TARGET
                penalty_total_volume = PENALTY_FACTOR * violation
                
            penalty_plastic_volume = 0
            if V_plastic > V_PLASTIC_ABSOLUTE_MAX:
                violation = V_plastic - V_PLASTIC_ABSOLUTE_MAX
                penalty_plastic_volume = PENALTY_FACTOR * 10 * violation
            
            # 2. Dự đoán Cường độ Nén (CS)
            x_scaled = scaler.transform(np.array([full_vector])) 
            pred_CS = model.predict(x_scaled)[0]
            delta_CS = abs(pred_CS - target_CS)
            
            # 3. Kết hợp Đa Mục tiêu
            total_fitness = (
                delta_CS + 
                (L1 * c_d_mass) - # Tối thiểu Xi măng (c_d_mass)
                (L2 * p_q_mass) + # Tối đa Nhựa (p_q_mass)
                penalty_total_volume + 
                penalty_plastic_volume
            )
            return total_fitness
    
    

    problem = MyProblem()
    # FOX.DevFOX là lớp con của BaseGA, nên import BaseGA chỉ để khai báo
    ga_solver = FOX.DevFOX(epoch=100, pop_size=30) 
    
    try:
        best_agent = ga_solver.solve(problem)
        best_solution = best_agent.solution
    except Exception as e:
        print(f"GA Solver error: {e}. This usually occurs when the search bounds are too narrow.")
        return {
            "Error": f"GA Solver Error: {e}",
            "Predicted_CS": None
        }

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

# ==== 5. Gradio interface ====
# Tổng số inputs là 1 (Target CS) + 14 Features = 15 inputs
# Chúng ta sẽ sử dụng DataFrame Output để hiển thị bảng vật liệu
output_components = [
    gr.Textbox(label="Predicted Concrete compressive strength (MPa)", min_width=200),
    gr.Dataframe(label="Optimized Materials", headers=["Component", "Value"], wrap=True)
]

# Tạo list input components (1 Target CS + 14 Features)
input_components = [gr.Number(label="Target Compressive Strength (MPa)")]
for feat in all_features:
    input_components.append(
        gr.Number(label=f"{name_map[feat]} ({feat})", value=None)
    )

def run_inverse(*args):
    try:
        target_CS = float(args[0]) 
        fixed_features = {}
        
        # Bắt đầu vòng lặp từ args[1:] (các features)
        for feat, val in zip(all_features, args[1:]): 
            
            # CẢI THIỆN QUAN TRỌNG:
            # 1. Kiểm tra không phải là None
            # 2. KIỂM TRA KHÁC 0: Chỉ coi là ràng buộc cố định nếu giá trị > 0 hoặc < 0.
            #    Nếu val == 0, chúng ta coi như người dùng không muốn cố định (để GA tối ưu).
            if val is not None and val != 0:
                fixed_features[feat] = val
        
        # Kiểm tra nếu Target CS quá lớn/nhỏ
        if target_CS < 10 or target_CS > 90:
             return (f"WARNING: Target CS {target_CS} MPa is outside the range of reliable data (10-90 MPa).", pd.DataFrame())


        # Chạy tối ưu hóa
        results = optimize_materials(target_CS, **fixed_features)
        
        if "Error" in results:
            return (results["Error"], pd.DataFrame())

        optimized_materials = results["Optimized Materials"]
        predicted_cs = results["Predicted_CS"]

        # Định dạng output thành DataFrame
        df = pd.DataFrame(optimized_materials.items(), columns=["Component", "Value"])
        
        # Trả về kết quả
        return f"CS Target: {target_CS} MPa | Predicted CS: {predicted_cs} MPa", df

    except Exception as e:
        # Bắt các lỗi khác như lỗi chuyển đổi kiểu dữ liệu
        return f"Unknow Error: {e}", pd.DataFrame()


demo = gr.Interface(
    fn=run_inverse,
    inputs=input_components,
    outputs=output_components,
    title="Recycled Plastic Waste Aggregate Concrete Inverse Design (GA-FOX)",
    description="Enter the target compressive strength (Target CS) and fix the components you want to keep unchanged. FOX (GA) will optimize the remaining 14 components.",
    
    # Ví dụ mẫu (15 giá trị: Target CS + 14 Features)
    examples=[
        # Target CS = 45 MPa, Curing Water (w) = 170.0, Xi măng (c_d) = 300.0
        [45.0, 300.0, None, None, None, None, None, None, 170.0, None, None, None, None, None, None], 
        # Target CS = 55 MPa, Slump = 80
        [55.0, None, None, None, None, None, None, None, None, None, None, None, 80, None, None]
    ]
)

demo.launch(inbrowser=True)