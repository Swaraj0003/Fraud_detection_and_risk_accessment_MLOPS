# Use official Python image
FROM python:3.8.10-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire app
COPY . .

# Run the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
