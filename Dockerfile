# Use an official Python runtime as a parent image (replace 3.11 with your desired version)
FROM python:3.13-slim

# Set work directory in the container
WORKDIR /app

# Install pip requirements (if requirements.txt exists)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Command to run your app (update as necessary, e.g., app.py or main.py)
CMD ["python", "main.py"]