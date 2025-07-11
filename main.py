# main.py

import joblib
from fastapi import FastAPI
from pydantic import BaseModel

# 1. Create a FastAPI app instance
app = FastAPI(title="Sentiment Analysis API", description="An API to predict sentiment of text.", version="1.0")

# 2. Load the trained model and class names
# These are loaded once when the application starts.
try:
    model = joblib.load('sentiment_model.joblib')
    class_names = joblib.load('class_names.joblib')
    print("Model and class names loaded successfully.")
except FileNotFoundError:
    print("Error: Model or class names file not found. Please run train.py first.")
    model = None
    class_names = None

# 3. Define the request body structure using Pydantic
# This ensures that any data sent to our endpoint conforms to this schema.
class TextInput(BaseModel):
    text: str

# 4. Define the response body structure
class SentimentResponse(BaseModel):
    prediction: str
    confidence: float
    text: str

# 5. Create the prediction endpoint
@app.post("/predict", response_model=SentimentResponse, tags=["Prediction"])
def predict_sentiment(request: TextInput):
    """
    Predicts the sentiment of a given text.
    - **text**: The input text to analyze.
    """
    if not model or not class_names:
        return {"error": "Model not loaded. Please check server logs."}

    # The input text from the request
    text_to_predict = request.text

    # Make predictions
    # model.predict() gives the class index (e.g., 0 or 1)
    # model.predict_proba() gives the probability for each class
    predicted_index = model.predict([text_to_predict])[0]
    prediction_probabilities = model.predict_proba([text_to_predict])[0]

    # Get the human-readable label and the confidence score
    predicted_class_name = class_names[predicted_index]
    confidence_score = prediction_probabilities[predicted_index]

    # Return the structured response
    return SentimentResponse(
        prediction=predicted_class_name,
        confidence=round(confidence_score, 4),
        text=text_to_predict
    )

# A simple root endpoint to confirm the API is running
@app.get("/", tags=["General"])
def read_root():
    return {"message": "Welcome to the Sentiment Analysis API. Go to /docs for more info."}