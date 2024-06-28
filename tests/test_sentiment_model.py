"""
Author: Onur Sasmaz
Copyright (c) 2024 Onur Sasmaz. All Rights Reserved.

"""
import pytest
from app.models.sentiment_model import SentimentModel
import os
import logging
import numpy as np

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture
def sample_data():
    """
    Fixture providing sample reviews and corresponding sentiments for testing.

    Returns:
    tuple: A tuple containing a list of sample reviews and a list of sentiments.
    """
    reviews = [
        "I love this product! It's amazing.", 
        "worst purchase I've ever made.", 
        "Absolutely fantastic!", 
        "Terrible quality, very disappointed.",
        "Great value for the price.", 
        "Not worth the money.", 
        "Awesome product but expensive", 
        "Worst product ever!"
    ]
    sentiments = ["positive", "negative", "positive", "negative", "positive", "negative", "positive", "negative"]
    return reviews, sentiments

@pytest.fixture
def sentiment_model():
    """
    Fixture providing an instance of SentimentModel for testing.

    Returns:
    SentimentModel: An instance of the SentimentModel class.
    """
    model = SentimentModel()
    return model

def test_train_model(sentiment_model, sample_data):
    """
    Test case to verify training of the SentimentModel.

    Args:
    sentiment_model (SentimentModel): An instance of the SentimentModel class.
    sample_data (tuple): A tuple containing a list of sample reviews and a list of sentiments.
    """
    reviews, sentiments = sample_data
    sentiment_model.train(reviews, sentiments)
    assert sentiment_model.model is not None

def test_predict(sentiment_model, sample_data):
    """
    Test case to verify prediction functionality of the SentimentModel.

    Args:
    sentiment_model (SentimentModel): An instance of the SentimentModel class.
    sample_data (tuple): A tuple containing a list of sample reviews and a list of sentiments.
    """
    reviews, sentiments = sample_data
    sentiment_model.train(reviews, sentiments)
    predictions = sentiment_model.predict(reviews)
    assert len(predictions) == len(reviews)
    assert all(pred in ["positive", "negative"] for pred in predictions)

def test_save_and_load_model(sentiment_model, sample_data, tmp_path):
    """
    Test case to verify saving and loading of the SentimentModel.

    Args:
    sentiment_model (SentimentModel): An instance of the SentimentModel class.
    sample_data (tuple): A tuple containing a list of sample reviews and a list of sentiments.
    tmp_path (py.path.LocalPath): pytest-provided temporary directory.

    """
    reviews, sentiments = sample_data
    sentiment_model.train(reviews, sentiments)

    model_path = os.path.join(tmp_path, "sentiment_model.joblib")
    encoder_path = os.path.join(tmp_path, "label_encoder.joblib")
    sentiment_model.save(model_path, encoder_path)

    new_model = SentimentModel()
    new_model.load(model_path, encoder_path)

    assert new_model.model is not None
    assert new_model.label_encoder is not None

    predictions = new_model.predict(reviews)
    assert len(predictions) == len(reviews)
    assert all(pred in ["positive", "negative"] for pred in predictions)

def test_basic_prediction_structure(sentiment_model, sample_data):
    """
    Test case to verify the basic structure of predictions.

    Args:
    sentiment_model (SentimentModel): An instance of the SentimentModel class.
    sample_data (tuple): A tuple containing a list of sample reviews and a list of sentiments.
    """
    reviews, sentiments = sample_data
    sentiment_model.train(reviews, sentiments)
    predictions = sentiment_model.predict(reviews)
    assert len(predictions) == len(reviews)
    assert isinstance(predictions, np.ndarray)

def test_training_accuracy(sentiment_model, sample_data):
    """
    Test case to verify the training accuracy of the SentimentModel.

    Args:
    sentiment_model (SentimentModel): An instance of the SentimentModel class.
    sample_data (tuple): A tuple containing a list of sample reviews and a list of sentiments.
    """
    reviews, sentiments = sample_data
    sentiment_model.train(reviews, sentiments)
    predictions = sentiment_model.predict(reviews)
    accuracy = sum(1 for pred, true in zip(predictions, sentiments) if pred == true) / len(sentiments)
    assert accuracy >= 0.5  # Ensure the model performs at least 50/50

def test_no_exceptions_on_empty_data(sentiment_model):
    """
    Test case to verify that no exceptions are raised on empty data.

    Args:
    sentiment_model (SentimentModel): An instance of the SentimentModel class.
    """
    reviews = []
    sentiments = []
    try:
        sentiment_model.train(reviews, sentiments)
        predictions = sentiment_model.predict(reviews)
    except Exception:
        pytest.fail("Model training or prediction raised an exception on empty data.")
