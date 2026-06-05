# test.py
import requests

BASE_URL = "http://localhost:8000"

queries = [
    "What is the difference between Stack and Queue?",
    "Explain time complexity of binary search",
    "How does merge sort work?",
    "What are the advantages of linked lists over arrays?",
]

for q in queries:
    print(f"\n{'='*60}")
    print(f"Q: {q}")
    print('='*60)
    res = requests.post(f"{BASE_URL}/query", json={"query": q})
    data = res.json()
    print(f"Answer: {data['answer']}")
    print(f"Sources: {data['sources']}")
    print(f"Chunks: {data['chunks_used']}/{data['chunks_retrieved']} used")