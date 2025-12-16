# 1. Sử dụng image Python 3.9 Slim (Linux cơ bản)
# Môi trường này sẽ chạy trên mọi máy chủ/Docker Desktop (kể cả Win 11)
FROM python:3.13-slim
# 2. Thiết lập thư mục làm việc trong container
WORKDIR /app

# 3. Copy requirements.txt và cài đặt dependencies
COPY requirements.txt requirements.txt
# Cài đặt thư viện
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy tất cả các file cần thiết
# Đảm bảo các file này nằm cùng thư mục với Dockerfile:
# app.py, scaler.pkl, xgb_woa_best_model.pkl, templates/, static/
COPY app.py .
COPY scaler.pkl .
COPY xgb_woa_best_model.pkl .
COPY templates templates/
COPY static static/

# 5. Tạo thư mục logs và thiết lập VOLUME
# VOLUME giúp log được lưu trữ bền vững (persistent)
RUN mkdir -p logs
VOLUME /app/logs

# 6. Mở cổng mà ứng dụng sẽ chạy (mặc định 5004)
EXPOSE 5004

# 7. Lệnh khởi chạy ứng dụng bằng Gunicorn
# Gunicorn là web server cho production, thay thế app.run()
# Lệnh này sẽ gọi app.py và khởi chạy biến app (ứng dụng Flask)
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:5004", "app:app"]
