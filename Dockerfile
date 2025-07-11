# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run main_cli.py when the container launches
CMD ["python", "main_cli.py", "--train-data", "ceyhan_normalize_veri.xlsx", "--test-data", "kuzey_ege_test_verisi.xlsx", "--models", "all"]
