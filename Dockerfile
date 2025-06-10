# Use the official Python 3.12 image
FROM python:3.12-slim

# Set environment variables to fix the math library issue and GDAL version
ENV CMAKE_ARGS="-DEIGEN_TEST_NO_MATH_LIB=ON" \
    GDAL_VERSION=3.9.1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    build-essential \
    cmake \
    libeigen3-dev \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt into the container
COPY requirements.txt requirements.txt
RUN python -m venv .venv && \
    .venv\Scripts\activate.bat && \
    # Install Python dependencies
    pip install -r requirements.txt

# Clone the macposts repository
RUN pip install ./macposts

# Expose any necessary ports (if applicable)
# EXPOSE 8000
