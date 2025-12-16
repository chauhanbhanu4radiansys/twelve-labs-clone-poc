from pinecone import Pinecone
import time
import os

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If dotenv is not available, manually parse .env file
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip().strip('"').strip("'")

# Initialize Pinecone with API key from .env file
api_key = os.getenv("PINECONE_API_KEY")
if not api_key:
    raise ValueError("PINECONE_API_KEY not found in .env file")

pc = Pinecone(api_key=api_key)

# List of indexes to reset
TARGET_INDEXES = [
    "text-search",
    "audio-search",
    "desc-search",
    "video-search"
]

def reset_specific_index(index_name):
    """
    Delete all vectors from a specific index.
    """
    try:
        print(f"\n{'='*60}")
        print(f"Processing index: {index_name}")
        print(f"{'='*60}")
        
        index = pc.Index(index_name)
        
        stats = index.describe_index_stats()
        total_vectors = stats.get('total_vector_count', 0)
        print(f"Current vectors: {total_vectors}")
        
        if total_vectors == 0:
            print(f"Index '{index_name}' is already empty.")
            return
        
        # Delete all vectors from all namespaces
        namespaces = stats.get('namespaces', {})
        
        if namespaces:
            for namespace in namespaces.keys():
                print(f"Deleting from namespace: '{namespace}'")
                index.delete(delete_all=True, namespace=namespace)
        else:
            print("Deleting from default namespace")
            index.delete(delete_all=True)
        
        # Wait a moment for deletion to propagate
        time.sleep(2)
        
        # Verify deletion
        stats = index.describe_index_stats()
        remaining = stats.get('total_vector_count', 0)
        print(f"✓ Reset complete. Remaining vectors: {remaining}")
        
    except Exception as e:
        print(f"✗ Error processing index '{index_name}': {str(e)}")

def reset_target_indexes():
    """
    Reset only the specified target indexes.
    """
    print(f"\nResetting {len(TARGET_INDEXES)} specific index(es):")
    for idx in TARGET_INDEXES:
        print(f"  - {idx}")
    
    for index_name in TARGET_INDEXES:
        reset_specific_index(index_name)
    
    print(f"\n{'='*60}")
    print("=== Reset complete ===")
    print(f"{'='*60}")

if __name__ == "__main__":
    reset_target_indexes()
