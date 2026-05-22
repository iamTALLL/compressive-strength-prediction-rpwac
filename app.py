import os

ENV = os.environ.get("APP_ENV", "lite")

if ENV == "full":
    from app_full import app
else:
    from app_lite import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)