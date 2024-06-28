"""
Author: Onur Sasmaz
Copyright (c) 2024 Onur Sasmaz. All Rights Reserved.

"""
from app.models.sentiment_model import SentimentModel

class SentimentService:
    """
    A service class for making predictions using a sentiment analysis model.

    Attributes:
    sentiment_model (SentimentModel): An instance of SentimentModel for sentiment prediction.
    """

    def __init__(self, model_path, encoder_path):
        """
        Initializes the SentimentService with a trained sentiment analysis model.

        Args:
        model_path (str): File path to the trained model (joblib format).
        encoder_path (str): File path to the label encoder (joblib format).
        """
        self.sentiment_model = SentimentModel()
        self.sentiment_model.load(model_path, encoder_path)

    def predict(self, reviews):
        """
        Predicts sentiment labels for given reviews.

        Args:
        reviews (list): List of review texts to predict sentiment for.

        Returns:
        list: Predicted sentiment labels for each review.
        """
        return self.sentiment_model.predict(reviews)
