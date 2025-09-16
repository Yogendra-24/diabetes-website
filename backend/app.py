from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()

# Load trained model
model_path = os.path.join(os.path.dirname(__file__), "model_pipeline.joblib")
model = joblib.load(model_path)

# Serve frontend (index.html and static files)
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/static", StaticFiles(directory=frontend_path), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index_file = os.path.join(frontend_path, "index.html")
    with open(index_file, "r", encoding="utf-8") as f:
        return f.read()

# Input data format
class InputData(BaseModel):
    age: float
    bmi: float
    hbA1c_level: float
    blood_glucose_level: float

@app.post("/predict")
def predict(data: InputData):
    X = np.array([[data.age, data.bmi, data.hbA1c_level, data.blood_glucose_level]])
    prediction = model.predict(X)
    return {"prediction": int(prediction[0])}

# CORS (in case frontend fetches API separately)
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
