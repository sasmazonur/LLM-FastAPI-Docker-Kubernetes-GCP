"""
Author: Onur Sasmaz
Copyright (c) 2024 Onur Sasmaz. All Rights Reserved.

"""
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentModel:
    """
    A class representing a sentiment analysis model using Logistic Regression and TF-IDF vectorization.

    Attributes:
    model (Pipeline): The trained sentiment analysis model pipeline.
    label_encoder (LabelEncoder): Encoder for sentiment labels.

    Methods:
    train(reviews, sentiments):
        Trains the sentiment analysis model using provided reviews and sentiments.

    save(model_path, encoder_path):
        Saves the trained model and label encoder to specified files.

    load(model_path, encoder_path):
        Loads a trained model and label encoder from specified files.

    predict(reviews):
        Predicts sentiment labels for input reviews using the trained model.

    _plot_grid_search_results(grid_search, param_grid):
        Plots grid search results for hyperparameter optimization.
    """

    def __init__(self):
        """
        Initializes a SentimentModel instance.

        Attributes:
        model (None or Pipeline): Initially set to None, will hold the trained model.
        label_encoder (LabelEncoder): Initializes a LabelEncoder instance for sentiment labels.
        """
        self.model = None
        self.label_encoder = LabelEncoder()

    def train(self, reviews, sentiments):
        """
        Trains the sentiment analysis model using provided reviews and sentiments.

        Args:
        reviews (list): A list of strings containing review texts.
        sentiments (list): A list of strings containing corresponding sentiment labels ('positive' or 'negative').

        Raises:
        ValueError: If there are fewer than 2 samples for training or if cross-validation cannot be performed due to insufficient data.

        """
        try:
            if len(reviews) < 2:
                raise ValueError("Not enough data to perform training. Need at least 2 samples.")
            
            sentiments = self.label_encoder.fit_transform(sentiments)
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000)),
                ('clf', LogisticRegression())
            ])
            
            param_grid = {
                'tfidf__max_features': [5000, 10000],
                'clf__C': [0.1, 1, 10]
            }

            X_train, X_test, y_train, y_test = train_test_split(reviews, sentiments, test_size=0.2, random_state=42)
            
            # Ensure n_splits is not greater than the number of samples in the smallest class
            unique, counts = np.unique(y_train, return_counts=True)
            min_class_samples = counts.min()
            n_splits = min(5, min_class_samples)
            if n_splits < 2:
                raise ValueError("Not enough data to perform cross-validation. Need at least 2 training samples per class.")
            
            grid_search = GridSearchCV(pipeline, param_grid, cv=n_splits, n_jobs=-1, verbose=2)
            grid_search.fit(X_train, y_train)
            
            self.model = grid_search.best_estimator_
            accuracy_train = self.model.score(X_train, y_train)
            accuracy_test = self.model.score(X_test, y_test)
            logger.info(f'Training accuracy: {accuracy_train}')
            logger.info(f'Test accuracy: {accuracy_test}')
            logger.info("Best parameters found during grid search: %s", grid_search.best_params_)
            self._plot_grid_search_results(grid_search, param_grid)
        except ValueError as e:
            logger.error("An error occurred during model training: %s", e)
            self.model = None
        except Exception as e:
            logger.error("An unexpected error occurred during model training: %s", e)
            self.model = None

    def save(self, model_path, encoder_path):
        """
        Saves the trained model and label encoder to specified files.

        Args:
        model_path (str): File path where the model should be saved.
        encoder_path (str): File path where the label encoder should be saved.

        """
        try:
            joblib.dump(self.model, model_path)
            joblib.dump(self.label_encoder, encoder_path)
            logger.info("Model and label encoder saved successfully.")
        except Exception as e:
            logger.error("An error occurred while saving the model and encoder: %s", e)

    def load(self, model_path, encoder_path):
        """
        Loads a trained model and label encoder from specified files.

        Args:
        model_path (str): File path from where the model should be loaded.
        encoder_path (str): File path from where the label encoder should be loaded.

        """
        try:
            self.model = joblib.load(model_path)
            self.label_encoder = joblib.load(encoder_path)
            logger.info("Model and label encoder loaded successfully.")
        except Exception as e:
            logger.error("An error occurred while loading the model and encoder: %s", e)

    def predict(self, reviews):
        """
        Predicts sentiment labels for input reviews using the trained model.

        Args:
        reviews (list): A list of strings containing review texts.

        Returns:
        list: A list of predicted sentiment labels ('positive' or 'negative') for each input review.

        """
        try:
            if self.model is None:
                raise ValueError("Model is not trained.")
            predictions = self.model.predict(reviews)
            logger.info(f'Predictions: {predictions}')
            return self.label_encoder.inverse_transform(predictions)
        except Exception as e:
            logger.error("An error occurred during prediction: %s", e)
            return []

    def _plot_grid_search_results(self, grid_search, param_grid):
        """
        Plots grid search results for hyperparameter optimization.

        Args:
        grid_search (GridSearchCV): The trained GridSearchCV object.
        param_grid (dict): Dictionary containing parameter grid for grid search.

        """
        try:
            results = grid_search.cv_results_
            scores = results['mean_test_score'].reshape(len(param_grid['clf__C']), len(param_grid['tfidf__max_features']))
            sns.set()
            plt.figure(figsize=(8, 6))
            ax = sns.heatmap(scores, annot=True, fmt=".3f", cmap='viridis',
                             xticklabels=param_grid['tfidf__max_features'], yticklabels=param_grid['clf__C'])
            ax.set_title('Grid Search Mean Test Scores')
            ax.set_xlabel('TF-IDF max_features')
            ax.set_ylabel('Logistic Regression C')
            plt.show()
            logger.info("Grid search results plotted successfully.")
        except Exception as e:
            logger.error("An error occurred while plotting grid search results: %s", e)
