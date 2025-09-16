import pandas as pd

# Đọc file gốc
df = pd.read_csv("Data1.csv")

# Lấy 10 dòng đầu tiên (chỉ lấy cột input, bỏ cột output nếu có)
X = df.iloc[:, :-1]  # giả sử cột cuối là output

# Lưu thành file CSV mới để test
X.to_csv("test_full.csv", index=False)
print("File test_batch.csv đã được tạo với 10 dòng đầu tiên.")
