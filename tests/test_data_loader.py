"""
Author: Onur Sasmaz
Copyright (c) 2024 Onur Sasmaz. All Rights Reserved.

"""
import pytest
import pandas as pd
from app.utils.data_loader import load_data, clean_text

def test_clean_text():
    """
    Tests the clean_text function to ensure correct text cleaning.

    """
    assert clean_text("<p>Test</p>") == "Test"
    assert clean_text("Hello, World!") == "Hello World"
    assert clean_text("") == ""
    assert clean_text("No <b>HTML</b> tags!") == "No HTML tags"
    assert clean_text("Multiple    spaces.") == "Multiple spaces"

def test_load_data():
    """
    Tests the load_data function to ensure correct loading and cleaning of data from a CSV file.

    """
    df = load_data('tests/test_files/test_reviews.csv')
    assert not df.empty
    assert 'review' in df.columns
    assert 'sentiment' in df.columns
    assert len(df) == 4  # Ensure all rows are loaded

    # Check if the reviews are cleaned
    assert df['review'].iloc[0] == "I love this product Its amazing"
    assert df['review'].iloc[1] == "This is the worst purchase Ive ever made"
    assert df['review'].iloc[2] == "Absolutely fantastic Exceeded my expectations"
    assert df['review'].iloc[3] == "Terrible quality very disappointed"

def test_load_data_missing_columns():
    """
    Tests the load_data function with a CSV file missing expected columns.

    """
    df = pd.DataFrame({"content": ["This is a test."]})
    df.to_csv('tests/test_files/test_missing_columns.csv', index=False)
    with pytest.raises(ValueError, match="Expected columns 'review' and 'sentiment' not found"):
        load_data('tests/test_files/test_missing_columns.csv')

def test_load_data_empty_file():
    """
    Tests the load_data function with an empty CSV file.

    """
    df = pd.DataFrame(columns=['review', 'sentiment'])
    df.to_csv('tests/test_files/test_empty.csv', index=False)
    loaded_df = load_data('tests/test_files/test_empty.csv')
    assert loaded_df.empty
