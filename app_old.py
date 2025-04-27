import os
import warnings
import json

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

from flask import Flask, render_template, jsonify, request, session
from src.helper import download_hugging_face_embeddings
from src.prompt import *
from langchain_pinecone import PineconeVectorStore
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

app = Flask(__name__)
app.secret_key = 'your-secret-key'  # Required for session

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY

embeddings = download_hugging_face_embeddings()

index_name = 'medichatbot'
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type='similarity', search_kwargs={"k": 3})

# Initialize Ollama with Mistral
llm = Ollama(
    model="mistral",
    temperature=0.3,
    base_url="http://localhost:11434"  # Default Ollama URL
)

qa_chain = RetrievalQA.from_chain_type(
    llm, retriever=retriever, chain_type_kwargs={"prompt": prompt}
)

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/get', methods=['GET','POST'])
def chat():
    msg = request.form['msg']
    history = request.form.get('history', '[]')  # Get history from request
    
    # Parse history if it's a string
    if isinstance(history, str):
        try:
            history = json.loads(history)
        except:
            history = []
    
    # Format conversation history
    conversation_context = ""
    for entry in history:
        if isinstance(entry, dict):
            conversation_context += f"User: {entry.get('user', '')}\n"
            conversation_context += f"Assistant: {entry.get('assistant', '')}\n"
    
    # Add current message to history
    current_history = history + [{"user": msg}]
    
    # Create enhanced query with conversation context
    enhanced_query = f"""Previous conversation:
{conversation_context}

Current question: {msg}"""
    
    print("Query with context:", enhanced_query)
    response = qa_chain.invoke({"query": enhanced_query})
    print("Response: ", response['result'])
    
    # Add assistant's response to history
    current_history[-1]["assistant"] = response['result']
    
    return jsonify({
        "response": str(response['result']),
        "history": current_history
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)