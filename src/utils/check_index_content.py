import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pinecone import Pinecone
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from src.helper import download_hugging_face_embeddings
import json

def check_index_content():
    """Check and display detailed information about Pinecone indexes and their content"""
    try:
        # Load environment variables
        load_dotenv()
        
        # Initialize Pinecone
        pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
        
        # Get list of all indexes
        indexes = pc.list_indexes()
        
        if not indexes:
            print("No indexes found in your Pinecone account.")
            return
        
        print("\n=== Pinecone Indexes Content Check ===")
        print(f"Total indexes found: {len(indexes)}")
        
        # Initialize embeddings for vector store
        print("\nInitializing embeddings...")
        embeddings = download_hugging_face_embeddings()
        
        # Test queries
        test_queries = [
            "What is Asthma?"
        ]
        
        for index in indexes:
            print(f"\n{'='*50}")
            print(f"Index Name: {index.name}")
            print(f"{'='*50}")
            
            try:
                # Create index instance
                index_obj = pc.Index(index.name)
                
                # Get index statistics
                stats = index_obj.describe_index_stats()
                print(f"Total Vectors: {stats['total_vector_count']}")
                print(f"Dimension: {stats['dimension']}")
                
                if stats['total_vector_count'] > 0:
                    print("\nNamespace Details:")
                    for namespace, ns_stats in stats['namespaces'].items():
                        print(f"  Namespace: {namespace}")
                        print(f"  Vector Count: {ns_stats['vector_count']}")
                    
                    # Initialize vector store
                    vector_store = PineconeVectorStore.from_existing_index(
                        index_name=index.name,
                        embedding=embeddings
                    )
                    
                    # Try each test query
                    for query in test_queries:
                        print(f"\nTrying query: '{query}'")
                        try:
                            results = vector_store.similarity_search(
                                query,
                                k=2  # Get 2 results
                            )
                            
                            if results:
                                print(f"\nFound {len(results)} results for query: '{query}'")
                                for i, doc in enumerate(results, 1):
                                    print(f"\nDocument {i}:")
                                    print(f"Content Preview: {doc.page_content[:300]}...")
                                    if hasattr(doc, 'metadata') and doc.metadata:
                                        print(f"Metadata: {json.dumps(doc.metadata, indent=2)}")
                            else:
                                print(f"\nNo matching documents found for query: '{query}'")
                                
                        except Exception as e:
                            print(f"Error fetching content for query '{query}': {str(e)}")
                else:
                    print("\nIndex is empty - no vectors found.")
                
            except Exception as e:
                print(f"Error accessing index: {str(e)}")
            
            print(f"\n{'-'*50}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise e

if __name__ == "__main__":
    check_index_content() 