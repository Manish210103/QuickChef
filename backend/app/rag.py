import os
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import numpy as np

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Global embedder instance (loaded once on startup)
embedder = None

def load_embedder():
    """Load embedder once and cache in memory"""
    global embedder
    if embedder is not None:
        return embedder
    
    print("Loading embedding model (this happens only once)...")
    # Use device="cpu" to avoid GPU memory issues on Railway
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    print("✓ Model loaded successfully")
    return embedder

def embed_text(text: str):
    """Generate 1024-dimensional embeddings with padding"""
    model = load_embedder()
    embedding = model.encode(text, convert_to_numpy=True)
    # Pad to 1024 dimensions to match existing index
    padding = np.zeros(1024 - len(embedding))
    return np.concatenate([embedding, padding]).tolist()

def get_pinecone_index():
    """Get or create Pinecone index 'quickchef' (1024-d, cosine)."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "quickchef-1024"
    # list indexes with compatibility across client versions
    listed = pc.list_indexes()
    try:
        if isinstance(listed, dict) and "indexes" in listed:
            existing = {ix["name"] for ix in listed["indexes"]}
        elif hasattr(listed, "indexes"):
            existing = {ix.name for ix in listed.indexes}
        else:
            existing = {getattr(ix, "name", ix.get("name")) for ix in listed}
    except Exception:
        existing = set()

    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=1024,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return pc.Index(index_name)

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