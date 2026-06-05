import urllib.request
import json
import time

url = "http://localhost:7777/api/generate"
payload = {
    "model": "hf.co/DavidAU/DeepSeek-MOE-4X8B-R1-Distill-Llama-3.1-Deep-Thinker-Uncensored-24B-GGUF:Q6_K",
    "prompt": "Return only the word READY.",
    "stream": False
}
data = json.dumps(payload).encode('utf-8')
req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})

start = time.time()
print("Calling model...")
try:
    with urllib.request.urlopen(req, timeout=120) as response:
        result = json.loads(response.read().decode('utf-8'))
        print(f"Time: {time.time() - start:.2f}s")
        print(result.get("response", ""))
except Exception as e:
    print(f"Error: {e}")
