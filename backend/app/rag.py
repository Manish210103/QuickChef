import os
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv
import numpy as np

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(text: str):
    """Generate 1024-dimensional embeddings with padding"""
    embedding = embedder.encode(text)
    # Pad to 1024 dimensions to match existing index
    padding = np.zeros(1024 - len(embedding))
    return np.concatenate([embedding, padding]).tolist()

def get_pinecone_index():
    """Get Pinecone index"""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index("quickchef")

def ingest_data(csv_path="data/Cleaned_Indian_Food_Dataset.csv"):
    """Ingest CSV data into Pinecone"""
    index = get_pinecone_index()
    df = pd.read_csv(csv_path)
    
    batch_size = 100
    vectors = []
    
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing recipes"):
        # Create text for embedding
        text = f"Recipe: {row['TranslatedRecipeName']} Ingredients: {row['Cleaned-Ingredients']} Instructions: {row['TranslatedInstructions']} Cuisine: {row['Cuisine']}"
        
        vectors.append({
            "id": f"recipe_{i}",
            "values": embed_text(text),
            "metadata": {
                "name": str(row['TranslatedRecipeName']),
                "ingredients": str(row['Cleaned-Ingredients']),
                "instructions": str(row['TranslatedInstructions']),
                "cuisine": str(row['Cuisine']),
                "total_time": int(row['TotalTimeInMins']) if pd.notna(row['TotalTimeInMins']) else 0,
                "url": str(row['URL']) if pd.notna(row['URL']) else "",
                "image_url": str(row['image-url']) if pd.notna(row['image-url']) else ""
            }
        })
        
        # Batch upsert
        if len(vectors) >= batch_size:
            index.upsert(vectors)
            vectors = []
    
    # Upsert remaining vectors
    if vectors:
        index.upsert(vectors)
    
    return {"status": "success", "message": f"Successfully ingested {len(df)} recipes"}

def query_rag(query: str, top_k: int = 3, filters: dict = None):
    """Search recipes"""
    index = get_pinecone_index()
    query_vector = embed_text(query)
    
    search_params = {
        "vector": query_vector,
        "top_k": top_k,
        "include_metadata": True
    }
    
    if filters:
        search_params["filter"] = filters
    
    results = index.query(**search_params)
    
    return [
        {
            "score": round(match["score"], 4),
            "recipe": {
                "name": match["metadata"]["name"],
                "ingredients": match["metadata"]["ingredients"],
                "instructions": match["metadata"]["instructions"],
                "cuisine": match["metadata"]["cuisine"],
                "total_time": match["metadata"]["total_time"],
                "url": match["metadata"]["url"],
                "image_url": match["metadata"]["image_url"]
            }
        }
        for match in results["matches"]
    ]

