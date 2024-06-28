"""
Author: Onur Sasmaz
Copyright (c) 2024 Onur Sasmaz. All Rights Reserved.

"""
from fastapi import FastAPI
from app.routers import sentiment

app = FastAPI()
"""
FastAPI application for sentiment analysis.

Attributes:
app (FastAPI): FastAPI instance for handling HTTP requests.

"""

app.include_router(sentiment.router, prefix="/sentiment", tags=["sentiment"])
"""
Includes the sentiment router for handling sentiment analysis endpoints.

Args:
    router (Router): Router instance for sentiment analysis.
    prefix (str): URL prefix for sentiment analysis endpoints.
    tags (list): List of tags for sentiment analysis endpoints.
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
"""
Runs the FastAPI application using Uvicorn server.

Args:
    app (FastAPI): FastAPI application instance.
    host (str): Host address to run the server (default: "0.0.0.0").
    port (int): Port number to run the server (default: 8080).
"""
