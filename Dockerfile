# Author: Onur Sasmaz
# Description: This Dockerfile sets up the environment and dependencies required.
#              
# Usage:
# - Build the Docker image:
#       docker build -t [image_name] .
# - Run the Docker container:
#       docker run -d --name [container_name] [image_name]
#
#
# Copyright (c) 2024 Onur Sasmaz. All Rights Reserved.

# Use an official Python runtime as a parent image
FROM python:3.11

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 8080

# Command to run the FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
