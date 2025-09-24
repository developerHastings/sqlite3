# main.py - Complete FastAPI application
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import logging
from typing import List, Optional



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI(
    title="Iris Classification API",
    description="A simple ML API for iris flower classification using FastAPI",
    version="1.0.0",
    docs_url="/docs",  # Enable Swagger UI documentation
    redoc_url="/redoc"  # Enable ReDoc documentation
)

# Define Pydantic models for request/response validation
class IrisFeatures(BaseModel):
    """Pydantic model for input feature validation"""
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    
    # Example values for documentation
    class Config:
        schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }

class PredictionResponse(BaseModel):
    """Pydantic model for prediction response"""
    prediction: int
    predicted_class: str
    confidence: float
    model_version: str

class HealthResponse(BaseModel):
    """Pydantic model for health check response"""
    status: str
    model_loaded: bool
    model_version: str

# Global variables for model and metadata
model = None
model_version = "v1.0"
classes = ["setosa", "versicolor", "virginica"]

@app.on_event("startup")
async def startup_event():
    """Initialize model when application starts"""
    global model
    try:
        # Load or train the model
        model = load_or_train_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def load_or_train_model():
    """Load existing model or train a new one"""
    try:
        # Try to load pre-trained model
        model = joblib.load("iris_model.joblib")
        logger.info("Loaded pre-trained model")
    except FileNotFoundError:
        # Train new model if not found
        logger.info("Training new model...")
        iris = load_iris()
        X, y = iris.data, iris.target
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train Random Forest classifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model trained with accuracy: {accuracy:.3f}")
        
        # Save the model
        joblib.dump(model, "iris_model.joblib")
        logger.info("Model saved successfully")
    
    return model

@app.get("/", response_model=dict)
async def root():
    """Root endpoint providing API information"""
    return {
        "message": "Iris Classification API",
        "version": model_version,
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/batch_predict"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for monitoring"""
    model_status = model is not None
    return HealthResponse(
        status="healthy" if model_status else "unhealthy",
        model_loaded=model_status,
        model_version=model_version
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(features: IrisFeatures):
    """
    Predict iris flower class for a single sample
    
    - **sepal_length**: Sepal length in cm
    - **sepal_width**: Sepal width in cm  
    - **petal_length**: Petal length in cm
    - **petal_width**: Petal width in cm
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert features to numpy array
        input_features = np.array([[
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        ]])
        
        # Make prediction
        prediction = model.predict(input_features)[0]
        probabilities = model.predict_proba(input_features)[0]
        confidence = probabilities[prediction]
        
        logger.info(f"Prediction made: class {prediction} with confidence {confidence:.3f}")
        
        return PredictionResponse(
            prediction=int(prediction),
            predicted_class=classes[prediction],
            confidence=float(confidence),
            model_version=model_version
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/batch_predict")
async def predict_batch(features_list: List[IrisFeatures]):
    """Predict iris flower classes for multiple samples"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert list of features to 2D array
        input_features = np.array([[
            f.sepal_length, f.sepal_width, f.petal_length, f.petal_width
        ] for f in features_list])
        
        # Make batch predictions
        predictions = model.predict(input_features)
        probabilities = model.predict_proba(input_features)
        
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            confidence = prob[pred]
            results.append({
                "sample_id": i,
                "prediction": int(pred),
                "predicted_class": classes[pred],
                "confidence": float(confidence)
            })
        
        logger.info(f"Batch prediction completed for {len(results)} samples")
        
        return {
            "predictions": results,
            "total_samples": len(results),
            "model_version": model_version
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/model_info")
async def get_model_info():
    """Get information about the trained model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": type(model).__name__,
        "model_version": model_version,
        "feature_names": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "class_names": classes,
        "n_features": model.n_features_in_ if hasattr(model, 'n_features_in_') else 4
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",  # Allow external connections
        port=8000,       # Default port
        reload=True      # Auto-reload during development
    )