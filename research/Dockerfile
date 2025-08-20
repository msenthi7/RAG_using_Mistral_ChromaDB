FROM python:3.11-slim

WORKDIR /app

# Copy dependency list first for better build caching
COPY requirements.txt .

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose Flask port
EXPOSE 8080

# Run the app
CMD ["python", "app.py"]