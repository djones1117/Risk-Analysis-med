# Use a slim Python image
FROM python:3.11-slim

# Work directory inside the container
WORKDIR /app

# Install system deps (optional)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project (including models/risk_model.joblib)
COPY . .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Expose the port your app will run on
EXPOSE 8000

# Start FastAPI with Uvicorn
# Render 
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
