import requests
import json

url = "http://127.0.0.1:8000/generate"
body = {
    "prompt": "China is",
    "max_tokens": 100
}

resp = requests.post(url, json=body, headers={"Authorization": "Bearer your_token_here"})
print("status_code:", resp.status_code)
print("text:", resp.text)

try:
    print(json.dumps(resp.json(), indent=4, ensure_ascii=False))
except Exception as e:
    print("json parse error:", e)
