"""
Author: Onur Sasmaz
Copyright (c) 2024 Onur Sasmaz. All Rights Reserved.

"""
import pandas as pd
import re
from bs4 import BeautifulSoup
from app.utils.logger import get_logger

logger = get_logger(__name__)

def clean_text(text):
    """
    Cleans the input text by removing HTML tags and non-alphabetic characters.

    Args:
    text (str): The input text to clean.

    Returns:
    str: The cleaned text.
    """
    try:
        soup = BeautifulSoup(text, features="html.parser")
        text = soup.get_text()
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
    except Exception as e:
        logger.error(f"Error cleaning text: {e}")
        return ""
    return text

def load_data(filepath: str):
    """
    Loads the data from a CSV file and applies text cleaning.

    Args:
    filepath (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The loaded and cleaned DataFrame.
    """
    try:
        df = pd.read_csv(filepath, quotechar='"', delimiter=',')
        logger.info("Columns in dataset: %s", df.columns.tolist())
        if 'review' not in df.columns or 'sentiment' not in df.columns:
            raise ValueError("Expected columns 'review' and 'sentiment' not found")
        df = df.dropna(subset=['review', 'sentiment'])
        df['review'] = df['review'].apply(clean_text)
    except ValueError as e:
        logger.error(f"Error loading data: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame(columns=['review', 'sentiment'])
    return df
