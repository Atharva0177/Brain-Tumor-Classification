from contextlib import asynccontextmanager
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, Response, UploadFile, status
from PIL import Image

from app.health import dependency_health
from app.inference.service import ProductionModelService

model_service = ProductionModelService()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    model_service.start()
    yield
    model_service.stop()


app = FastAPI(title="Brain Tumor Classification API", version="0.1.0", lifespan=lifespan)


@app.get("/health", tags=["system"])
def health(response: Response) -> dict:
    """Return application and dependency readiness."""
    result = dependency_health()
    model_status = model_service.status()
    result["model"] = model_status
    if not model_status["loaded"]:
        result["status"] = "unavailable"
    if result["status"] != "ok":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return result


@app.get("/health/live", tags=["system"])
def liveness() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model-info", tags=["model"])
def model_info() -> dict:
    try:
        return model_service.info()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/predict", tags=["prediction"])
async def predict(file: UploadFile = File(...)) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload must be an image")
    try:
        payload = await file.read()
        image = Image.open(BytesIO(payload))
        image.verify()
        image = Image.open(BytesIO(payload)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc
    try:
        return model_service.predict(image)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
