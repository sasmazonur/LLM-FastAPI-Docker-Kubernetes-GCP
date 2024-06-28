"""
Author: Onur Sasmaz
Copyright (c) 2024 Onur Sasmaz. All Rights Reserved.

"""
from fastapi import APIRouter
from app.schemas.sentiment import ReviewRequest, SentimentResponse
from app.services.sentiment_service import SentimentService

router = APIRouter()
model_path = 'app/models/sentiment_model.joblib'
encoder_path = 'app/models/label_encoder.joblib'
sentiment_service = SentimentService(model_path, encoder_path)

@router.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(request: ReviewRequest):
    """
    Endpoint to predict sentiment for a list of reviews.

    Args:
    request (ReviewRequest): Request body containing a list of reviews.

    Returns:
    SentimentResponse: Response body containing a list of predicted sentiments.
    """
    sentiments = sentiment_service.predict(request.reviews)
    return SentimentResponse(sentiments=sentiments.tolist())
