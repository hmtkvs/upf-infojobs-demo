# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libffi-dev \
    libnacl-dev \
    python3-dev \
    build-essential \
    && apt-get -y install tesseract-ocr \ # required for pytesseract \
    && apt-get -y install ffmpeg libsm6 libxext6 # required for opencv

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /usr/src/app

# Copy only the requirements.txt first to leverage Docker cache
COPY ./requirements.txt /app/requirements.txt

# Create a virtual environment and install dependencies
RUN python -m venv venv && \
    . venv/bin/activate && \
    pip install --upgrade pip && \
    pip install --upgrade pip setuptools && \
    pip install -r requirements.txt


# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8501 available to the world outside this container
EXPOSE 8501

COPY . /app

# Run app.py when the container launches
CMD ["/app/venv/bin/streamlit", "run", "final_streamlit.py"]
