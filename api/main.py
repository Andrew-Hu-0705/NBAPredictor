import hashlib
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from api.logging_conf import get_logger, log_event
from api.registry import load_production_model
from api.schemas import HealthResponse, PredictRequest, PredictResponse, ShapFactor
from predict import load_artifacts, load_features, predict_game

logger = get_logger()

ARTIFACTS = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        from_registry = None
        try:
            from_registry = load_production_model()
        except Exception as exc:
            log_event(logger, "startup: MLflow registry load failed, falling back to local files",
                       event="startup", ok=False, source="mlflow", error=str(exc))

        if from_registry is not None:
            model, explainer, feature_cols, source = from_registry
        else:
            model, explainer, feature_cols = load_artifacts()
            source = "local joblib files"

        ARTIFACTS["model"] = model
        ARTIFACTS["explainer"] = explainer
        ARTIFACTS["feature_cols"] = feature_cols
        ARTIFACTS["features_df"] = load_features()
        ARTIFACTS["model_source"] = source
        log_event(logger, "startup: artifacts loaded", event="startup", ok=True, source=source)
    except Exception as exc:
        log_event(logger, "startup: failed to load artifacts", event="startup", ok=False, error=str(exc))
    yield
    ARTIFACTS.clear()


app = FastAPI(title="NBA Game Outcome Predictor", version="1.0.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health():
    model_loaded = ARTIFACTS.get("model") is not None and ARTIFACTS.get("explainer") is not None
    features_loaded = ARTIFACTS.get("features_df") is not None
    status = "ok" if model_loaded and features_loaded else "degraded"
    return HealthResponse(
        status=status, model_loaded=model_loaded, features_loaded=features_loaded,
        model_source=ARTIFACTS.get("model_source", "none"),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if ARTIFACTS.get("model") is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded")

    input_hash = hashlib.sha256(f"{req.home_team}|{req.away_team}".encode()).hexdigest()[:16]
    start = time.perf_counter()

    try:
        result = predict_game(
            req.home_team,
            req.away_team,
            model=ARTIFACTS["model"],
            explainer=ARTIFACTS["explainer"],
            feature_cols=ARTIFACTS["feature_cols"],
            features_df=ARTIFACTS["features_df"],
        )
    except ValueError as exc:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        log_event(
            logger, "prediction request failed", event="prediction", status="error",
            input_hash=input_hash, home_team=req.home_team, away_team=req.away_team,
            latency_ms=latency_ms, error=str(exc),
        )
        raise HTTPException(status_code=404, detail=str(exc))

    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    pairs = sorted(
        zip(result["shap_values"], result["feature_cols"], result["feature_values"]),
        key=lambda x: abs(x[0]),
        reverse=True,
    )
    shap_explanation = [
        ShapFactor(
            feature=feat,
            value=float(val),
            shap_value=float(shap_val),
            favors="home" if shap_val > 0 else "away",
        )
        for shap_val, feat, val in pairs
    ]

    response = PredictResponse(
        home_team=result["home_team"],
        away_team=result["away_team"],
        home_win_prob=result["home_win_prob"],
        away_win_prob=result["away_win_prob"],
        predicted_winner=result["predicted_winner"],
        confidence=result["confidence"],
        shap_explanation=shap_explanation,
    )

    log_event(
        logger, "prediction request served", event="prediction", status="ok",
        input_hash=input_hash, home_team=req.home_team, away_team=req.away_team,
        latency_ms=latency_ms, home_win_prob=result["home_win_prob"],
        predicted_winner=result["predicted_winner"],
    )

    return response
