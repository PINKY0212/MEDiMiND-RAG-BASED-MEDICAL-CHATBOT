import os
import warnings

# Filter out LangSmith warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langsmith")
# Filter out HuggingFace Hub FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
# Filter out LangChain deprecation warning
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")

# Completely disable LangSmith and all related warnings
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_TRACING"] = "false"
os.environ["LANGCHAIN_PROJECT"] = ""
os.environ["LANGCHAIN_SESSION"] = ""
os.environ["LANGCHAIN_CALLBACKS"] = "false"
os.environ["LANGCHAIN_HANDLER"] = "false"

from flask import Flask
from src.config import FLASK_SECRET_KEY
from src.models import initialize_models
from src.routes import setup_routes

def create_app():
    """Create and configure the Flask application"""
    app = Flask(__name__)
    app.secret_key = FLASK_SECRET_KEY
    
    # Initialize models
    models = initialize_models()
    
    # Setup routes
    setup_routes(app, models)
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=8080, use_reloader=False)