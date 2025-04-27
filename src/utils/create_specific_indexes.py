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

def create_specific_index(pc, embeddings, folder_name, data_path):
    """Create index for a specific folder"""
    print(f"\nProcessing {folder_name}...")
    
    # Create index name based on folder name
    index_name = f"medichatbot-{folder_name.lower().replace('_', '-')}"
    
    # Load and process documents
    folder_path = os.path.join(data_path, folder_name)
    text_chunks = text_split(load_pdf_file(folder_path))
    
    if not text_chunks:
        print(f"No documents found in {folder_name}")
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
    
    print(f"Completed processing {folder_name}!")

def main():
    try:
        # Initialize with proper multiprocessing setup
        set_start_method('spawn', force=True)
        
        print("Initializing Pinecone...")
        load_dotenv()
        pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
        
        print("Downloading embeddings...")
        embeddings = download_hugging_face_embeddings()
        
        # Define data path
        data_path = 'Data'
        
        # List of specific folders to process
        # folders_to_process = ['drugs_info', 'guideline']
        folders_to_process = ['guidelines']
        
        # Process each specified folder
        for folder in folders_to_process:
            create_specific_index(pc, embeddings, folder, data_path)
        
        print("\nAll operations completed successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    return 0

if __name__ == '__main__':
    exit(main()) 