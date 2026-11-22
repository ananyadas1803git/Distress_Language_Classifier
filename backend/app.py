from contextlib import asynccontextmanager
from pathlib import Path
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.schemas import (
    PredictRequest,
    PredictResponse,
    ModelBenchmark,
    ModelInfo,
    ExamplePost,
    AbstentionPoint,
    HealthResponse,
)
from backend.service import service, SAMPLE_POSTS


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize models on startup
    service.initialize()
    yield


app = FastAPI(
    title="Distress Language Classifier API",
    description="Responsible NLP API for detecting distress-related language with calibrated uncertainty and abstention.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for local React development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse, tags=["System"])
def get_health() -> HealthResponse:
    """Returns system status and loaded models."""
    return HealthResponse(
        status="healthy" if service.is_initialized else "initializing",
        models_loaded=len(service.models),
        vectorizer_fitted=service.tfidf is not None and service.tfidf.is_fitted,
        message="Classifier backend is ready for inference."
    )


@app.get("/api/models", response_model=List[ModelInfo], tags=["Models"])
def list_models() -> List[ModelInfo]:
    """Lists available baseline models and architectural properties."""
    return service.get_models_info()


@app.post("/api/predict", response_model=PredictResponse, tags=["Inference"])
def predict_distress(req: PredictRequest) -> PredictResponse:
    """Classifies an online post, evaluates confidence against threshold, and returns token attributions."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text field cannot be empty.")
    return service.predict(
        text=req.text,
        model_name=req.model_name,
        confidence_threshold=req.confidence_threshold,
    )


@app.get("/api/metrics", response_model=List[ModelBenchmark], tags=["Evaluation"])
def get_benchmark_metrics() -> List[ModelBenchmark]:
    """Returns comparative metrics (Accuracy, F1, ROC-AUC, Brier score, ECE) across all models."""
    return service.benchmark_metrics


@app.get("/api/abstention-sweep", response_model=List[AbstentionPoint], tags=["Uncertainty"])
def get_abstention_sweep() -> List[AbstentionPoint]:
    """Returns coverage versus selective accuracy and risk across confidence thresholds."""
    return service.abstention_sweep


@app.get("/api/examples", response_model=List[ExamplePost], tags=["Data"])
def get_example_posts() -> List[ExamplePost]:
    """Returns curated sample posts for 1-click testing."""
    return SAMPLE_POSTS


# Mount built frontend static files if distribution directory exists
frontend_dist = Path("frontend/dist")
if frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        file_path = frontend_dist / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(frontend_dist / "index.html")
