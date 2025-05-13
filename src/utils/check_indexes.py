from pinecone import Pinecone
from dotenv import load_dotenv
import os

def delete_specific_indexes(pc, index_names):
    """Delete specific indexes from Pinecone"""
    try:
        for index_name in index_names:
            if index_name in pc.list_indexes().names():
                print(f"\nDeleting index: {index_name}")
                pc.delete_index(index_name)
                print(f"Successfully deleted index: {index_name}")
            else:
                print(f"\nIndex {index_name} not found")
    except Exception as e:
        print(f"Error occurred while deleting indexes: {str(e)}")

def check_pinecone_indexes():
    """Check and display information about existing Pinecone indexes"""
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
        
        print("\n=== Pinecone Indexes Information ===")
        print(f"Total indexes found: {len(indexes)}")
        print("\nDetailed Information:")
        
        for index in indexes:
            print(f"\nIndex Name: {index.name}")
            print(f"Dimension: {index.dimension}")
            print(f"Metric: {index.metric}")
            print(f"Host: {index.host}")
            print(f"Spec: {index.spec}")
            
            # Get index statistics
            index_stats = pc.describe_index(index.name)
            print(f"Status: {index_stats.status}")
            print(f"Created at: {index_stats.created_at}")
            
            # Get vector count if index is ready
            if index_stats.status == 'ready':
                index_obj = pc.Index(index.name)
                stats = index_obj.describe_index_stats()
                print(f"Total Vectors: {stats.total_vector_count}")
                print(f"Total Namespaces: {len(stats.namespaces)}")
                if stats.namespaces:
                    print("Namespace Details:")
                    for ns, details in stats.namespaces.items():
                        print(f"  - {ns}: {details.vector_count} vectors")
            
            print("-" * 50)
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Initialize Pinecone
    pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
    print("Pinecone API key - ", pc)
    
    # # Define indexes to delete 'medichatbot-research-papers', 'medichatbot-guidelines', 'medichatbot-drugs-info', 'medichatbot-hybrid'
    # indexes_to_delete = [ 'medichatbot-drugs-info']
    
    # # Delete specific indexes
    # print("\n=== Deleting Specific Indexes ===")
    # delete_specific_indexes(pc, indexes_to_delete)
    
    # Show remaining indexes after deletion
    print("\n=== Total Indexes in Pinecone ===")
    check_pinecone_indexes() 