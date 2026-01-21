import requests
import json

url = 'http://127.0.0.1:5000/predict'
data = {
    'alcohol': 14.23,
    'flavanoids': 3.06,
    'color_intensity': 5.64,
    'hue': 1.04,
    'od280/od315_of_diluted_wines': 3.92,
    'proline': 1065
}

try:
    print(f"Sending request to {url}...")
    response = requests.post(url, json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response Body: {response.text}")
    
    if response.status_code == 200:
        result = response.json()
        print("\nTest PASSED")
        print(f"Predicted: {result.get('prediction')}")
        print(f"Confidence: {result.get('confidence')}")
    else:
        print("\nTest FAILED")

except Exception as e:
    print(f"Connection failed: {e}")
