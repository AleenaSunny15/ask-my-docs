import os
import requests
from dotenv import load_dotenv

load_dotenv()
headers = {"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"}
r = requests.get("https://api.groq.com/openai/v1/models", headers=headers, timeout=30)

print("Status:", r.status_code)
if r.status_code == 200:
    for m in r.json()["data"]:
        print(m["id"])
else:
    print(r.text)