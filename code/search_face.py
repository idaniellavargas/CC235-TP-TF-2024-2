from pinecone import Pinecone
import time
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv('PINECONE_KEY'))
index_name = "students-faces-embeddings-512"

while not pc.describe_index(index_name).status['ready']:
    time.sleep(1)

index = pc.Index(index_name)

def query_face_embedding(embedding: np.ndarray, k: int = 5, test: bool = False):
    embedding = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
    
    query_results = index.query(
        namespace="test" if test else "prod",
        vector=embedding,
        top_k=k,
        include_values=False,
        include_metadata=True
    )

    return query_results
