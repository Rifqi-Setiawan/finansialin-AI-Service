FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV/Pillow if needed
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Start script will be provided by docker-compose
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
