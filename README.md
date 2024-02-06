
# Installation and Usage Guide

## Prerequisites

- Docker installed on your machine. For installation instructions, visit the [official Docker documentation](https://docs.docker.com/get-docker/).

## Running the API Server

1. **Pull the Docker Image**

   Open a terminal and run the following command to pull the latest version of the `tensordt/c4_inference` Docker image:

   ```bash
   docker pull tensordt/c4_inference:latest
   ```

2. **Start the API Server**

   Execute the following command to run the Docker container and start the API server:

   ```bash
   docker run -p 5000:5000 tensordt/c4_inference:latest
   ```

   This command maps port 5000 of the Docker container to port 5000 on your host machine, allowing you to access the API at `http://127.0.0.1:5000/predict`.

## Accessing the API

To access the API, you can use the following Python script as an example. This script sends a sample request to the API and prints the response.

### Sample Python Client

1. **Prepare Your Environment**

   Ensure you have Python and the `requests` library installed. You can install `requests` using pip:

   ```bash
   pip install requests
   ```

2. **Execute the Python Script**

   Copy the script below into a file, for example, `api_client.py`, and run it with Python:

   ```python
   import requests

   # URL of the Flask API
   url = 'http://127.0.0.1:5000/predict'

   # Sample input data
   sample_data = {
       'PART_NUMBER': '89-32KHZ#S08',
       'TRANSACTION_QTY': 10804,
       'ACT_CT': 23.2013,
       'ROUTE': 'P41421_WIP',
       'HOLD_DAYS': 0.0,
       'PLAN_CT': 0.692132082,
       'PKG_CODE': 'W20#H2',
       'TESTER': 'MAV1064',
       'BU': "nan",
       'TESTERTYPE': 'MAV1',
       'EXCLUDE_FLAG': 'NONTESTFLOW',
       'PLAN_CT_100PERC': 0.408357928,
       'TESTSTEPCNT': 3.0,
       'PLAN_CT_100PERC2H1D': 1.658357928,
       'WASHOLD': 0,
       'HOLDREASONS': "nan",
       'PLAN_CT_100PERC1D': 1.408357928,
       'OEEFACTOR': 0.59,
       'OEEFACTORSRC': 'CPT_HIST',
       'DIETYPE': 'MO78A-0A',
       'PACKAGETYPE': 'SOIC (W)',
       'DOWNTIME_DAYS': 0.877430556,
       'QUEUE_DAYS': "nan",
       'HOLIDAY_HRS': "nan"
   }

   # Make a POST request to the API
   response = requests.post(url, json=sample_data)

   if response.status_code == 200:
       prediction = response.json()['prediction']
       print(f"Predicted Output: {prediction}")
   else:
       print("Failed to get a prediction from the API:", response.status_code, response.text)
   ```

3. **Run the Script**

   Execute the script in your terminal:

   ```bash
   python api_client.py
   ```

This script sends a POST request to your Flask API and prints out the response, which includes the predicted output based on the sample input data.
