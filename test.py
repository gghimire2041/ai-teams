import requests
import json

url = "http://localhost:1234/v1/chat/completions"
payload = {
    "model": "openai/gpt-oss-20b",
    "messages": [{"role": "user", "content": "Hello from Python"}],
    "temperature": 0.7,
    "max_tokens": 50
}

response = requests.post(url, json=payload)
print(f"Status: {response.status_code}")
print(f"Response: {response.text}")