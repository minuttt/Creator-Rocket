import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv

from db.database import engine, Base
from api.routes import router

# Load Environment Variables
load_dotenv()

# Logging Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("creatorrocket")

# Create SQLite Database Tables
Base.metadata.create_all(bind=engine)

# Background Scheduler Setup
scheduler = BackgroundScheduler()

def scheduled_job():
    from services.scheduler import fetch_scheduled_stats
    fetch_scheduled_stats()

# Run every 30 minutes
scheduler.add_job(scheduled_job, 'interval', minutes=30)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup Logic
    logger.info("Starting CreatorRocket API...")
    
    # Train base ML model if it doesn't exist
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "xgboost_model.pkl")
    if not os.path.exists(model_path):
        logger.info("Training initial XGBoost model...")
        import numpy as np
        from xgboost import XGBClassifier
        from sklearn.model_selection import train_test_split
        from joblib import dump
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        rng = np.random.RandomState(42)
        n = 5000
        X = rng.uniform(0, 100, (n, 8))
        y = (X[:,0] * 0.4 + X[:,3] * 0.6 > 50).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = XGBClassifier(n_estimators=100, max_depth=3, verbosity=0)
        model.fit(X_train, y_train)
        dump(model, model_path)
        logger.info("Base model trained and saved.")

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
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)