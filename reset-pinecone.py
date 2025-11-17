from pinecone import Pinecone
import time

# Initialize Pinecone
pc = Pinecone(api_key="pcsk_68fghf_5w37zk5oyZwVJdLx645ueUtfrLyGSb22yvdT2qyPRxBGG2BLrUvDhaK8shRPugW")

def reset_all_indexes():
    """
    Delete all vectors from all indexes in your Pinecone account.
    """
    # Get list of all indexes
    indexes = pc.list_indexes()
    
    if not indexes:
        print("No indexes found.")
        return
    
    print(f"Found {len(indexes)} index(es)")
    
    for idx in indexes:
        index_name = idx['name']
        print(f"\nProcessing index: {index_name}")
        
        try:
            # Connect to the index
            index = pc.Index(index_name)
            
            # Get index stats to check current vector count
            stats = index.describe_index_stats()
            total_vectors = stats.get('total_vector_count', 0)
            print(f"  Current vectors: {total_vectors}")
            
            if total_vectors == 0:
                print(f"  Index '{index_name}' is already empty.")
                continue
            
            # Delete all vectors from all namespaces
            namespaces = stats.get('namespaces', {})
            
            if namespaces:
                # Delete from each namespace
                for namespace in namespaces.keys():
                    print(f"  Deleting all vectors from namespace: '{namespace}'")
                    index.delete(delete_all=True, namespace=namespace)
            else:
                # Delete from default namespace
                print(f"  Deleting all vectors from default namespace")
                index.delete(delete_all=True)
            
            # Wait a moment for deletion to propagate
            time.sleep(2)
            
            # Verify deletion
            stats = index.describe_index_stats()
            remaining = stats.get('total_vector_count', 0)
            print(f"  ✓ Completed. Remaining vectors: {remaining}")
            
        except Exception as e:
            print(f"  ✗ Error processing index '{index_name}': {str(e)}")
    
    print("\n=== Reset complete ===")

def reset_specific_index(index_name):
    """
    Delete all vectors from a specific index.
    """
    try:
        index = pc.Index(index_name)
        
        stats = index.describe_index_stats()
        total_vectors = stats.get('total_vector_count', 0)
        print(f"Index '{index_name}' has {total_vectors} vectors")
        
        if total_vectors == 0:
            print("Index is already empty.")
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
        
        time.sleep(2)
        stats = index.describe_index_stats()
        print(f"✓ Reset complete. Remaining vectors: {stats.get('total_vector_count', 0)}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    # Option 1: Reset all indexes
    reset_all_indexes()
    
    # Option 2: Reset specific index (uncomment to use)
    # reset_specific_index("your-index-name")