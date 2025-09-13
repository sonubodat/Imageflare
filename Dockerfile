# Use an official lightweight Python image as a parent image
FROM python:3.9-slim

# Set the working directory in the container to /app
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir ensures the image is smaller
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code (app.py and the src folder) into the container at /app
COPY . .

# Make port 8501 available to the world outside this container
# This is the default port for Streamlit
EXPOSE 8501

# Define a healthcheck to ensure the app is running
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run app.py when the container launches
# The --server.address=0.0.0.0 flag is essential for the container to be accessible
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]