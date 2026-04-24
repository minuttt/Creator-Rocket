import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
from sqlalchemy import inspect, text

from db.database import engine, Base
from api.routes import router

# Load Environment Variables
load_dotenv()

# Logging Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("creatorrocket")

# Create SQLite Database Tables
Base.metadata.create_all(bind=engine)

def ensure_database_schema():
    inspector = inspect(engine)
    if "creators" not in inspector.get_table_names():
        return

    creator_columns = {col["name"] for col in inspector.get_columns("creators")}
    with engine.begin() as conn:
        if "profile_picture_url" not in creator_columns:
            conn.execute(text("ALTER TABLE creators ADD COLUMN profile_picture_url VARCHAR"))
            logger.info("Added creators.profile_picture_url column")

# Background Scheduler Setup
scheduler = BackgroundScheduler()

FEATURE_NAMES = [
    "velocity_7d", "velocity_30d", "acceleration", "engagement_rate",
    "virality_score", "consistency_score", "niche_momentum", "audience_quality"
]

def ensure_prediction_model():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "xgboost_model.pkl")
    sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_creators.csv")

    model_is_usable = False
    if os.path.exists(model_path):
        try:
            import joblib
            joblib.load(model_path)
            model_is_usable = True
        except Exception as exc:
            logger.warning("Existing model is unusable and will be replaced: %s", exc)

    if model_is_usable:
        return

    if not os.path.exists(sample_path):
        logger.warning("Sample training dataset not found; continuing without a trained model.")
        return

    logger.info("Training calibrated XGBoost model from sample_creators.csv ...")
    import pandas as pd
    from joblib import dump
    from xgboost import XGBClassifier

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    df = pd.read_csv(sample_path)
    X = df[FEATURE_NAMES].astype("float32")
    y = df["exploded"].astype("int32")

    model = XGBClassifier(
        n_estimators=160,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        verbosity=0,
        random_state=42
    )
    model.fit(X, y)
    dump(model, model_path)
    logger.info("Calibrated breakout model saved to disk.")

def scheduled_job():
    from services.scheduler import fetch_scheduled_stats
    fetch_scheduled_stats()

# Run every 30 minutes
scheduler.add_job(scheduled_job, 'interval', minutes=30)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup Logic
    logger.info("Starting CreatorRocket API...")

    ensure_database_schema()
    ensure_prediction_model()

    scheduler.start()
    logger.info("Background scheduler started. Tracking creators will update every 30 mins.")
    yield
    
    # Shutdown Logic
    scheduler.shutdown()

app = FastAPI(title="CreatorRocket API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"]
)

app.include_router(router)

# Serve Frontend UI
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    frontend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "index.html")
    if os.path.exists(frontend_path):
        with open(frontend_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>frontend/index.html not found</h1>"

if __name__ == "__main__":
    import uvicorn
    import os
    # Render provides a PORT environment variable. We use 8000 as a fallback for your laptop.
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
