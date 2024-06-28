# Sentiment Analysis Deployment on Google Cloud Platform

## Overview

This project demonstrates how to build, deploy, and manage a sentiment analysis application using Scikit-Learn, FastAPI, Docker, and Kubernetes on Google Cloud Platform (GCP).

The sentiment analysis model classifies text reviews into positive or negative sentiments. We will containerize the application using Docker and deploy it on GCP with Kubernetes, ensuring efficient resource management.

## Prerequisites

Before you start, make sure you have the following:
- A Google account for accessing GCP services
- Basic understanding of cloud computing and Kubernetes concepts
- Google Cloud SDK installed on your local machine
- Python 3.6+ installed
- Docker installed
- Postman or a similar API testing tool

## Project Structure

Here is the high-level overview of the project structure:

```
sentiment-analysis-project/
├── app/
│ ├── models/
│ ├── routers/
│ ├── schemas/
│ ├── services/
│ ├── utils/
│ ├── main.py
├── data/
├── tests/
├── train_model.py
```

## Steps to Follow

### 1. Data Loading and Cleaning
- Load and clean the review data from a CSV file.
- Implement text cleaning functions to prepare the data for training.

### 2. Model Training
- Train a logistic regression model using TF-IDF vectorization.
- Save the trained model and label encoder.

### 3. API Creation
- Create an API using FastAPI to handle sentiment prediction requests.
- Define routes and request/response schemas.

### 4. Containerization with Docker
- Create a Dockerfile to containerize the FastAPI application.
- Build and run the Docker image locally to ensure everything works as expected.

### 5. Deployment on GCP with Kubernetes
- Push the Docker image to Docker Hub.
- Create a Kubernetes cluster on GCP.
- Deploy the application using Kubernetes deployment and service YAML files.

### 6. Testing and Validation
- Test the API endpoints locally using Postman.
- Validate the model's predictions and performance metrics.

### 7. Clean Up Resources
- Delete Kubernetes resources and clusters to avoid unnecessary charges.
- Remove Docker images and containers from your local machine and Docker Hub.

## Follow the Complete Guide

For a detailed step-by-step guide, follow the complete tutorial on Medium: [Deploy LLM Sentiment Analysis on Google Cloud Platform: Scikit-Learn, FastAPI, Docker, and Kubernetes Tutorial 2024](https://medium.com)

## Conclusion

By following this tutorial, you will gain hands-on experience in building and deploying a sentiment analysis system using modern web technologies and cloud infrastructure. This comprehensive guide equips you with the skills to integrate machine learning capabilities into your projects, ensuring scalability and maintainability.

Happy coding!

---