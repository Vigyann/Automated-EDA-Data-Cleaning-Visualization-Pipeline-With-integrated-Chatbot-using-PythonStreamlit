from dotenv import load_dotenv
import os
import requests

# Load API key from .env
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

model_id = "google/flan-t5-base"  # heavy model, may fail on free tier
API_URL = f"https://api-inference.huggingface.co/models/{model_id}"
headers = {"Authorization": f"Bearer {hf_token}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    print("Status Code:", response.status_code)   # 👈 debug
    print("Raw Response:", response.text[:500])   # 👈 debug
    try:
        return response.json()
    except Exception as e:
        return {"error": str(e), "text": response.text}

if __name__ == "__main__":
    output = query({"inputs": "Explain bar chart in simple terms."})
    print("Output:", output)
