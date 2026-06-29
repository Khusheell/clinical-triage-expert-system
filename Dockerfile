# Dockerfile

# Step 1: Start with a Python base image
FROM python:3.10-slim

# Step 2: Install the system-level dependency for pyswip: SWI-Prolog
RUN apt-get update && apt-get install -y swi-prolog && \
    rm -rf /var/lib/apt/lists/*

# Step 3: Set the working directory inside the container
WORKDIR /app

# Step 4: Copy the requirements file and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy your *entire* project into the container
COPY . .

# Step 6: Expose the port your Flask app runs on
EXPOSE 5000

# Step 7: Define the command to run your application
# This runs your 'app' variable from your 'app.py' file.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]