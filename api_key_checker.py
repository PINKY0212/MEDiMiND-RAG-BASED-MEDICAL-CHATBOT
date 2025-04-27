import requests
import os

HUGGINGFACEHUB_API_TOKEN = os.environ.get('HUGGINGFACEHUB_API_TOKEN')
os.environ['HUGGINGFACEHUB_API_TOKEN'] = HUGGINGFACEHUB_API_TOKEN

def check_huggingface_api_key():
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    response = requests.get(API_URL, headers=headers)
    
    if response.status_code == 200:
        print("✅ HuggingFace API key is valid!")
    else:
        print(f"❌ HuggingFace API key error: {response.status_code} - {response.text}")

    import requests
headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
response = requests.get("https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2", headers=headers)
print(response.status_code)  # Should return 200


if __name__ == '__main__':
    check_huggingface_api_key()