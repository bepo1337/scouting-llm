from pymilvus import connections, utility

def drop_all_collections():
    # Connect to the Milvus server
    connections.connect("default", host="localhost", port="19530")
    
    # List all collections
    collections = utility.list_collections()
    
    # Drop each collection
    for collection in collections:
        print(f"Dropping collection: {collection}")
        utility.drop_collection(collection)
    
    print("All collections have been dropped.")

if __name__ == "__main__":
    drop_all_collections()
