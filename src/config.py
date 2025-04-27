import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define categories and their indices
CATEGORIES = {
    'research_papers': 'medichatbot-research-papers',
    'guidelines': 'medichatbot-guidelines',
    'drug_info': 'medichatbot-drugs-info',
    'hybrid': 'medichatbot-hybrid'
}

# Pinecone configuration
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY

# Flask configuration
FLASK_SECRET_KEY = 'your-secret-key'

# Ollama configuration
OLLAMA_CONFIG = {
    "model": "mistral",
    "temperature": 0.3,
    "base_url": "http://localhost:11434"
}

# Search configuration
SEARCH_CONFIG = {
    "search_type": 'similarity',
    "search_kwargs": {"k": 3}
}