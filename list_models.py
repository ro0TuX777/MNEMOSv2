import urllib.request
import json
url = "http://localhost:7777/api/tags"
try:
    with urllib.request.urlopen(url) as response:
        data = json.loads(response.read().decode('utf-8'))
        for model in data.get('models', []):
            print(model['name'])
except Exception as e:
    print(f"Error: {e}")
