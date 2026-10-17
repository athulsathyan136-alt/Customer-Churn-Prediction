from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (PROJECT_ROOT/"models"/"customer_churn_pipeline.joblib")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found {MODEL_PATH}")

model_package = joblib.load(MODEL_PATH)

model = model_package["model"]
feature_names = model_package["feature_names"]


app = FastAPI(
    title='Customer Churn Prediction API',
    description='Predict customer churm probability',
    version='1.0.0'

)

class Customer(BaseModel):

    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

@app.get('/')
def home():
    return{
      "message": "Customer Churn Prediction API",
        "docs": "/docs",  
    }

@app.get("/health")
def health():
    return{"status": "healthy", "model_loaded": model is not None

    }

@app.post("/predict")
def predict(customer:Customer):
    try:
        customer_data = pd.DataFrame([customer.model_dump()],columns=feature_names,)

        probability = float(model.predict_proba(customer_data)[0,1])

        prediction = int(model.predict(customer_data)[0])

        return {
            "churn_proability":round(probability,4),
            "Prediction":("Churn risk" if prediction == 1 else "Likely to Stay"),
        }

    except(ValueError,TypeError) as error:
        raise HTTPException(status_code=422,detail=f"Count not process customer data : {error}",)from error