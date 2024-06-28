"""
Author: Onur Sasmaz
Copyright (c) 2024 Onur Sasmaz. All Rights Reserved.

"""
from pydantic import BaseModel
from typing import List

class ReviewRequest(BaseModel):
    """
    Request model for sending a list of reviews.

    Attributes:
    reviews (List[str]): List of review texts.
    """
    reviews: List[str]

class SentimentResponse(BaseModel):
    """
    Response model for returning a list of sentiments.

    Attributes:
    sentiments (List[str]): Predicted sentiments corresponding to the input reviews.
    """
    sentiments: List[str]
