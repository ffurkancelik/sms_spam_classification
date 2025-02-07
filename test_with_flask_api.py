import requests

url = "http://127.0.0.1:5000/predict"
data = {"text": "I wont. So wat's wit the guys"}

response = requests.post(url, json=data)
print("SMS: ", data['text'])
print("Prediction: ", response.json()['prediction'])