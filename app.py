import time
import logging
import numpy as np

from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel

from sklearn.linear_model import LinearRegression

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST
)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI application
app = FastAPI(title="Student Result Prediction API")

# Train a demonstration model
X = np.array([
    [60, 50, 55, 45],
    [70, 65, 70, 60],
    [80, 75, 80, 75],
    [90, 85, 88, 80],
    [95, 90, 95, 90],
    [50, 40, 45, 35]
])

y = np.array([50, 65, 78, 85, 95, 40])

model = LinearRegression()
model.fit(X, y)

# Metrics
HTTP_REQUESTS = Counter(
    "student_api_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"]
)

HTTP_LATENCY = Histogram(
    "student_api_http_request_duration_seconds",
    "HTTP request duration",
    ["method", "path"]
)

PREDICTIONS = Counter(
    "student_api_predictions_total",
    "Total prediction attempts",
    ["result"]
)

MODEL_LOADED = Gauge(
    "student_api_model_loaded",
    "Whether model is loaded"
)

MODEL_LOADED.set(1)

# Input schema
class StudentResultInput(BaseModel):
    attendance: float
    internal_marks: float
    assignment_marks: float
    previous_score: float

# Monitoring middleware
@app.middleware("http")
async def monitor_requests(request, call_next):

    start = time.perf_counter()

    response = await call_next(request)

    duration = time.perf_counter() - start

    path = request.url.path
    method = request.method

    if path != "/metrics":

        HTTP_REQUESTS.labels(
            method=method,
            path=path,
            status=str(response.status_code)
        ).inc()

        HTTP_LATENCY.labels(
            method=method,
            path=path
        ).observe(duration)

    return response

# Health endpoint
@app.get("/health")
def health():
    return {"status": "healthy"}

# Prediction endpoint
@app.post("/predict")
def predict(data: StudentResultInput):

    try:

        input_data = np.array([[
            data.attendance,
            data.internal_marks,
            data.assignment_marks,
            data.previous_score
        ]])

        prediction = model.predict(input_data)

        PREDICTIONS.labels(result="success").inc()

        logger.info("Prediction successful")

        return {
            "predicted_score": round(float(prediction[0]), 2)
        }

    except Exception:

        PREDICTIONS.labels(result="error").inc()

        logger.exception("Prediction failed")

        raise

# Prometheus metrics endpoint
@app.get("/metrics")
def metrics():

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
