# C:\Users\Asus\Downloads\Medical-Chat-Bot-main\Medical-Chat-Bot-main\src\store_index.py

from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os
import warnings
from multiprocessing import set_start_method

# Configure environment
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
warnings.filterwarnings("ignore", category=UserWarning)

def process_subfolder(pc, embeddings, subfolder_name, data_path):
    """Process a single subfolder and create its index"""
    print(f"\nProcessing {subfolder_name}...")
    
    # Create index name based on subfolder, ensuring it's lowercase and uses only alphanumeric characters and hyphens
    index_name = f"medichatbot-{subfolder_name.lower().replace('_', '-')}"
    
    # Load and process documents
    subfolder_path = os.path.join(data_path, subfolder_name)
    text_chunks = text_split(load_pdf_file(subfolder_path))
    
    if not text_chunks:
        print(f"No documents found in {subfolder_name}")
        return
    
    # Check if index exists
    if index_name in pc.list_indexes().names():
        print(f"Using existing index '{index_name}'...")
    else:
        print(f"Creating new index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=384,  # gte-small embedding dimension
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    
    # Upload documents with batch processing
    print("Uploading documents (this may take a while)...")
    batch_size = 100
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i + batch_size]
        PineconeVectorStore.from_documents(
            documents=batch,
            index_name=index_name,
            embedding=embeddings
        )
        print(f"Uploaded batch {i//batch_size + 1}/{(len(text_chunks)//batch_size)+1}")
    
    print(f"Completed processing {subfolder_name}!")

def main():
    try:
        # Initialize with proper multiprocessing setup
        set_start_method('spawn', force=True)
        
        print("Initializing Pinecone...")
        load_dotenv()
        pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
        
        print("Downloading embeddings...")
        embeddings = download_hugging_face_embeddings()
        
        # Define data path and get subfolders
        data_path = 'Data'
        subfolders = [d for d in os.listdir(data_path) 
                     if os.path.isdir(os.path.join(data_path, d))]
        
        if not subfolders:
            raise ValueError("No subfolders found in Data directory")
        
        # Process each subfolder
        for subfolder in subfolders:
            process_subfolder(pc, embeddings, subfolder, data_path)
        
        print("\nAll operations completed successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    return 0

if __name__ == '__main__':
    exit(main())