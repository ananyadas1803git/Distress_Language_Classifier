from typing import List, Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Post text to classify")
    model_name: str = Field(
        default="TF-IDF + Logistic Regression",
        description="Name of the model to use for classification"
    )
    confidence_threshold: float = Field(
        default=0.70,
        ge=0.50,
        le=0.99,
        description="Confidence threshold for selective abstention"
    )


class TokenAttribution(BaseModel):
    token: str
    weight: float
    direction: str  # 'distress' or 'neutral'


class PredictResponse(BaseModel):
    text: str
    cleaned_text: str
    model_name: str
    probability: float
    confidence: float
    confidence_threshold: float
    is_abstained: bool
    predicted_label: int
    decision: str
    top_tokens: List[TokenAttribution]
    ethical_notice: str


class ModelBenchmark(BaseModel):
    model: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    brier_score: float
    ece: float


class ModelInfo(BaseModel):
    id: str
    name: str
    algorithm: str
    description: str
    strengths: str
    is_calibrated: bool


class ExamplePost(BaseModel):
    id: str
    title: str
    category: str
    text: str


class AbstentionPoint(BaseModel):
    confidence_threshold: float
    coverage: float
    abstention_rate: float
    retained_count: int
    abstained_count: int
    selective_accuracy: float
    selective_f1: float
    selective_risk: float


class HealthResponse(BaseModel):
    status: str
    models_loaded: int
    vectorizer_fitted: bool
    message: str
