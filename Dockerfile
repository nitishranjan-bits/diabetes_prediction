# Use the official Python 3.9 image
FROM python:3.9

# Set the working directory
WORKDIR /opt/diabetes_prediction

# Copy the requirements file and install the dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /opt/diabetes_prediction/

# Expose the ports for Flask application and MLflow server
EXPOSE 5000
EXPOSE 5001

# Command to run your Flask application
CMD ["python3", "src/app.py"]
