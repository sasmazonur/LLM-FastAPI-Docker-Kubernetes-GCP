"""
Author: Onur Sasmaz
Copyright (c) 2024 Onur Sasmaz. All Rights Reserved.

"""
from app.utils.data_loader import load_data
from app.models.sentiment_model import SentimentModel
from app.utils.visualization import plot_data_distribution, visualize_model_performance
from app.utils.logger import get_logger

logger = get_logger(__name__)

if __name__ == "__main__":
    data_path = 'data/reviews.csv'
    model_path = 'app/models/sentiment_model.joblib'
    encoder_path = 'app/models/label_encoder.joblib'

    logger.info("Loading data...")
    df = load_data(data_path)

    # Plot data distribution
    logger.info("Plotting data distribution...")
    plot_data_distribution(df)

    reviews = df['review'].values
    sentiments = df['sentiment'].values

    logger.info("Training the model...")
    sentiment_model = SentimentModel()
    sentiment_model.train(reviews, sentiments)
    sentiment_model.save(model_path, encoder_path)

    # Visualize model performance
    logger.info("Visualizing model performance...")
    visualize_model_performance(df, model_path, encoder_path)
