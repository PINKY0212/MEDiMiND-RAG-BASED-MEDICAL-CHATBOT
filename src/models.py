import warnings
warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")

from langchain_pinecone import PineconeVectorStore
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from src.config import CATEGORIES, OLLAMA_CONFIG, SEARCH_CONFIG
from src.helper import download_hugging_face_embeddings
from src.prompt import prompt

def initialize_models():
    """Initialize all models and vector stores"""
    try:
        embeddings = download_hugging_face_embeddings()
        
        # Initialize vector stores for each category
        vector_stores = {}
        for category, index_name in CATEGORIES.items():
            try:
                vector_stores[category] = PineconeVectorStore.from_existing_index(
                    index_name=index_name,
                    embedding=embeddings
                )
                print(f"Successfully loaded index: {index_name}")
            except Exception as e:
                print(f"Error loading index {index_name}: {str(e)}")
                raise Exception(f"Failed to initialize vector store for {category}")
        
        # Initialize Ollama with Mistral
        llm = OllamaLLM(
            model=OLLAMA_CONFIG['model'],
            temperature=OLLAMA_CONFIG['temperature'],
            base_url=OLLAMA_CONFIG['base_url']
        )
        
        return {
            'llm': llm,
            'vector_stores': vector_stores
        }
    except Exception as e:
        print(f"Error initializing models: {str(e)}")
        raise

    # Initialize retrievers
    retrievers = {}
    for category, vector_store in vector_stores.items():
        if vector_store:
            retrievers[category] = vector_store.as_retriever(
                search_type=SEARCH_CONFIG['search_type'],
                search_kwargs=SEARCH_CONFIG['search_kwargs']
            )
    
    # Create QA chains for each category
    qa_chains = {}
    for category, retriever in retrievers.items():
        if retriever:
            qa_chains[category] = RetrievalQA.from_chain_type(
                llm,
                retriever=retriever,
                chain_type_kwargs={"prompt": prompt}
            )
    
    return {
        'llm': llm,
        'vector_stores': vector_stores,
        'retrievers': retrievers,
        'qa_chains': qa_chains
    } 