"""Land Acquisition Delay Predictor - FastAPI Application Entrypoint.

Problem Statement: SIH26017 - AI-Powered Predictive Analytics Platform
for Early Detection and Mitigation of Infrastructure Land Acquisition Delays.
"""

from contextlib import asynccontextmanager
import logging
from pathlib import Path
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import joblib
import shap

from backend.app.routers.dashboard import router as dashboard_router
from backend.app.routers.ingestion import router as ingestion_router
from backend.app.routers.predictions import router as predictions_router
from backend.app.routers.recommendations import router as recommendations_router

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("landsight.main")

# Determine model directory path (robust to execution from workspace root or backend dir)
BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "ml" / "models"
if not MODELS_DIR.exists():
    MODELS_DIR = Path("ml/models").resolve()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for eager preloading of ML models into app.state."""
    logger.info(f"[*] Starting Land Acquisition Delay Predictor API service. Loading ML models from: {MODELS_DIR}...")
    start_time = time.time()

    # 1. Load Multiclass Risk Classifier
    classifier_path = MODELS_DIR / "risk_classifier.joblib"
    if classifier_path.exists():
        app.state.risk_classifier = joblib.load(classifier_path)
        logger.info(f"[+] Loaded Risk Classifier from {classifier_path.name}")
        # Pre-initialize TreeExplainer for fast in-memory SHAP attributions
        try:
            app.state.tree_explainer = shap.TreeExplainer(app.state.risk_classifier["model"])
            logger.info("[+] Pre-initialized SHAP TreeExplainer in app state.")
        except Exception as e:
            logger.warning(f"[-] Could not pre-initialize SHAP TreeExplainer: {e}")
            app.state.tree_explainer = None
    else:
        logger.warning(f"[-] Missing classifier bundle: {classifier_path}")
        app.state.risk_classifier = None
        app.state.tree_explainer = None

    # 2. Load Survival Analysis Models (Compensation & Possession)
    comp_path = MODELS_DIR / "survival_compensation.joblib"
    if comp_path.exists():
        app.state.survival_compensation = joblib.load(comp_path)
        logger.info(f"[+] Loaded Compensation Survival Model from {comp_path.name}")
    else:
        logger.warning(f"[-] Missing compensation survival bundle: {comp_path}")
        app.state.survival_compensation = None

    poss_path = MODELS_DIR / "survival_possession.joblib"
    if poss_path.exists():
        app.state.survival_possession = joblib.load(poss_path)
        logger.info(f"[+] Loaded Possession Survival Model from {poss_path.name}")
    else:
        logger.warning(f"[-] Missing possession survival bundle: {poss_path}")
        app.state.survival_possession = None

    # 3. Load Delay Propagation Stage-Pair Models
    prop_path = MODELS_DIR / "delay_propagation_models.joblib"
    if prop_path.exists():
        app.state.delay_propagation_models = joblib.load(prop_path)
        logger.info(f"[+] Loaded Delay Propagation Models from {prop_path.name}")
    else:
        logger.warning(f"[-] Missing delay propagation models: {prop_path}")
        app.state.delay_propagation_models = None

    elapsed = time.time() - start_time
    logger.info(f"[+] All ML models successfully loaded into app.state in {elapsed:.2f}s.")

    yield

    logger.info("[*] Shutting down Land Acquisition Delay Predictor API service...")


app = FastAPI(
    title="Land Acquisition Delay Predictor",
    description=(
        "Production backend for SIH26017: Early Detection and Mitigation of Infrastructure "
        "Land Acquisition Delays with RFCTLARR 2013 Milestone Tracking and Explainable AI."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure Cross-Origin Resource Sharing (CORS) for Vite frontend dev server
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Adds X-Process-Time response header to track latency."""
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000.0
    response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"
    return response


# Register Routers
app.include_router(predictions_router, prefix="/api", tags=["ML Predictions & Simulation"])
app.include_router(dashboard_router, prefix="/api", tags=["Dashboard & Spatial Analytics"])
app.include_router(recommendations_router, prefix="/api", tags=["AI Recommendations & Alerts"])
app.include_router(ingestion_router, prefix="/api", tags=["Project Ingestion"])



@app.get("/", tags=["System"])
def root():
    """Returns basic service metadata and API health."""
    return {
        "service": "Land Acquisition Delay Predictor API",
        "problem_statement": "SIH26017",
        "status": "online",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/api/health", tags=["System"])
def health_check():
    """System health check and loaded model telemetry."""
    return {
        "status": "healthy",
        "models_loaded": {
            "risk_classifier": app.state.risk_classifier is not None,
            "survival_compensation": app.state.survival_compensation is not None,
            "survival_possession": app.state.survival_possession is not None,
            "delay_propagation_models": app.state.delay_propagation_models is not None,
            "tree_explainer": app.state.tree_explainer is not None,
        },
    }
