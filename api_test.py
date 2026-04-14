import pandas as pd
import requests
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def test_api():
    url = "http://127.0.0.1:5000/predict"
    
    try:
        # Read the sample ECG data
        logging.info("Reading sample_ecg.csv...")
        df = pd.read_csv("sample_ecg.csv")
        
        # Take the first row as a list
        sample_input = df.iloc[0].tolist()
        
        payload = {
            "input": sample_input
        }
        
        logging.info(f"Sending POST request to {url}")
        response = requests.post(url, json=payload)
        
        logging.info(f"Response Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            logging.info("Prediction Result:")
            logging.info(json.dumps(result, indent=2))
        else:
            logging.error(f"Error Response: {response.text}")
            
    except Exception as e:
        logging.error(f"Failed to test API: {e}")

if __name__ == "__main__":
    test_api()
