import requests
import json

# URL of your Flask API
url = 'http://127.0.0.1:5000/predict'

# Sample input data (matching the expected input structure of your API)
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

# Make a POST request to the Flask API
response = requests.post(url, json=sample_data)

# Assuming the API returns a JSON with a key 'prediction'
if response.status_code == 200:
    prediction = response.json()['prediction']
    print(f"Predicted Output: {prediction}")
else:
    print("Failed to get a prediction from the API:", response.status_code, response.text)
