from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()

# ----------------------
# Visitor counter setup
# ----------------------
VISITOR_FILE = os.path.join(os.path.dirname(__file__), "visitors.txt")

# Initialize visitor count if file doesn't exist
if not os.path.exists(VISITOR_FILE):
    with open(VISITOR_FILE, "w") as f:
        f.write("0")

def increment_visitor():
    with open(VISITOR_FILE, "r+") as f:
        count = int(f.read())
        count += 1
        f.seek(0)
        f.write(str(count))
        f.truncate()
    return count

def get_visitor_count():
    with open(VISITOR_FILE, "r") as f:
        return int(f.read())

# ----------------------
# Load trained model
# ----------------------
model_path = os.path.join(os.path.dirname(__file__), "model_pipeline.joblib")
model = joblib.load(model_path)

# ----------------------
# Serve frontend
# ----------------------
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/static", StaticFiles(directory=frontend_path), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    visitor_count = increment_visitor()  # Increment visitor on each visit
    index_file = os.path.join(frontend_path, "index.html")
    with open(index_file, "r", encoding="utf-8") as f:
        html = f.read()
    # Inject visitor count (initial) into HTML
    html = html.replace("{{visitor_count}}", str(visitor_count))
    return html

# ----------------------
# Prediction API
# ----------------------
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

# ----------------------
# Visitor API
# ----------------------
@app.get("/visitors")
def visitors():
    count = get_visitor_count()
    return JSONResponse(content={"count": count})

# ----------------------
# CORS
# ----------------------
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
