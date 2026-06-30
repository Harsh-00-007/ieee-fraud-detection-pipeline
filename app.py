from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# 1. Initialize the web server
app = FastAPI(title="Fraud Detection ML API", version="2.0")

# 2. Load the Model (Happens ONLY ONCE when the server boots up)
# Make sure this path matches where your V2 model is saved!
MODEL_PATH = os.path.join('models', 'xgb_fraud_model.joblib') 

try:
    print("Loading XGBoost Model into RAM...")
    model = joblib.load(MODEL_PATH)
    expected_cols = model.feature_names_in_
    print("✅ Model loaded successfully. Server ready.")
except Exception as e:
    print(f"❌ CRITICAL WARNING: Could not load model. {e}")
    model = None

# 3. Data Schema (The Security Guard)
# This tells the internet: "Send me a JSON object containing a dictionary of features."
class Transaction(BaseModel):
    features: dict

# 4. The Prediction Endpoint (The actual bridge to the AI)
@app.post("/predict")
def predict_fraud(transaction: Transaction):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is offline.")
    
    # Create a base dataframe from the input
    df = pd.DataFrame([transaction.features])
    
    # Find which columns are missing
    missing_cols = [c for c in expected_cols if c not in df.columns]
    
    # Add all missing columns at once using a dictionary to avoid fragmentation
    if missing_cols:
        missing_data = {col: -999 for col in missing_cols}
        df = df.assign(**missing_data)
        
    # Reorder to match model's expected input
    df = df[expected_cols]
    
    # Predict
    probability = float(model.predict_proba(df)[0][1])
    is_fraud = bool(probability > 0.85)
    
    return {
        "status": "success",
        "fraud_probability": round(probability, 4),
        "action_required": "BLOCK_TRANSACTION" if is_fraud else "ALLOW_CHECKOUT",
    }