import requests
import os
import pinecone

HUGGINGFACEHUB_API_TOKEN = os.environ.get('HUGGINGFACEHUB_API_TOKEN')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.environ.get('PINECONE_ENVIRONMENT', 'gcp-starter')  # default to gcp-starter if not specified

os.environ['HUGGINGFACEHUB_API_TOKEN'] = HUGGINGFACEHUB_API_TOKEN

def check_huggingface_api_key():
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
    # Using the HuggingFace API info endpoint which is more reliable for testing
    API_URL = "https://huggingface.co/api/whoami"
    response = requests.get(API_URL, headers=headers)
    
    if response.status_code == 200:
        print("✅ HuggingFace API key is valid!")
        print(f"Connected as: {response.json().get('name', 'Unknown user')}")
    else:
        print(f"❌ HuggingFace API key error: {response.status_code} - {response.text}")

def check_pinecone_api_key():
    if not PINECONE_API_KEY:
        print("❌ Pinecone API key not found in environment variables")
        return
    
    try:
        # Initialize Pinecone
        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
        # List indexes to verify API key
        indexes = pinecone.list_indexes()
        print("✅ Pinecone API key is valid!")
        print(f"Available indexes: {indexes}")
    except Exception as e:
        print(f"❌ Pinecone API key error: {str(e)}")

if __name__ == '__main__':
    print("\nChecking HuggingFace API key...")
    check_huggingface_api_key()
    
    print("\nChecking Pinecone API key...")
    check_pinecone_api_key()

