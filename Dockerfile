# Base image Python 3.13 Slim
FROM python:3.13-slim

WORKDIR /app

# Cài dependencies trước (tận dụng Docker cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ source (app_lite.py, app_full.py, models, templates, static...)
COPY . .

# Tạo thư mục logs persistent
RUN mkdir -p logs
VOLUME /app/logs

# APP_ENV=full  → HuggingFace (app_full.py)
# APP_ENV=lite  → Render free  (app_lite.py)
ENV APP_ENV=full

# HF Spaces yêu cầu port 7860
# Render / local dùng PORT env var (mặc định 5004)
EXPOSE 7860
EXPOSE 5004

# Entrypoint script để chọn đúng app + port theo môi trường
CMD ["sh", "-c", "\
  if [ \"$APP_ENV\" = 'full' ]; then APP_MODULE='app_full:app'; else APP_MODULE='app_lite:app'; fi && \
  PORT=${PORT:-5004} && \
  exec gunicorn --workers 4 --bind 0.0.0.0:$PORT $APP_MODULE \
"]