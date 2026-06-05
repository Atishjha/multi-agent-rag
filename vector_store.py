# vector_store.py
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

def get_collection(name: str = "documents"):
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}
    )