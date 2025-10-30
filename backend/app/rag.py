import os
import json
from pinecone import Pinecone
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY)

def embed_text(text: str):
    """
    Generate embeddings using a simple hash-based approach for lightweight deployment.
    For production, consider using an embedding API like OpenAI, Cohere, or Voyage AI.
    """
    # Simple deterministic embedding based on text content
    # This creates a 1024-dimensional vector from text
    from hashlib import sha256
    
    # Create multiple hash variations for better distribution
    hashes = []
    for i in range(32):  # 32 hashes * 32 bytes = 1024 dimensions
        hash_input = f"{text}_{i}".encode()
        hash_bytes = sha256(hash_input).digest()
        # Convert bytes to normalized floats
        for byte in hash_bytes:
            hashes.append((byte / 255.0) - 0.5)  # Normalize to [-0.5, 0.5]
    
    return hashes[:1024]  # Ensure exactly 1024 dimensions

# Alternative: Use Groq's embedding model (if available)
# def embed_text(text: str):
#     """Generate embeddings using Groq API"""
#     try:
#         response = groq_client.embeddings.create(
#             model="your-embedding-model",
#             input=text
#         )
#         embedding = response.data[0].embedding
#         # Pad or truncate to 1024 dimensions
#         if len(embedding) < 1024:
#             embedding.extend([0.0] * (1024 - len(embedding)))
#         return embedding[:1024]
#     except Exception as e:
#         print(f"Embedding error: {e}")
#         return [0.0] * 1024

def get_pinecone_index():
    """Get Pinecone index"""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index("quickchef")

def ingest_data(csv_path="data/Cleaned_Indian_Food_Dataset.csv"):
    """Ingest CSV data into Pinecone without heavy dependencies"""
    import csv
    
    index = get_pinecone_index()
    
    batch_size = 100
    vectors = []
    total_recipes = 0
    
    print(f"Starting ingestion from {csv_path}...")
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for i, row in enumerate(reader):
            # Create text for embedding
            text = f"Recipe: {row.get('TranslatedRecipeName', '')} Ingredients: {row.get('Cleaned-Ingredients', '')} Instructions: {row.get('TranslatedInstructions', '')} Cuisine: {row.get('Cuisine', '')}"
            
            # Parse total time safely
            try:
                total_time = int(float(row.get('TotalTimeInMins', 0)))
            except (ValueError, TypeError):
                total_time = 0
            
            vectors.append({
                "id": f"recipe_{i}",
                "values": embed_text(text),
                "metadata": {
                    "name": str(row.get('TranslatedRecipeName', '')),
                    "ingredients": str(row.get('Cleaned-Ingredients', '')),
                    "instructions": str(row.get('TranslatedInstructions', '')),
                    "cuisine": str(row.get('Cuisine', '')),
                    "total_time": total_time,
                    "url": str(row.get('URL', '')),
                    "image_url": str(row.get('image-url', ''))
                }
            })
            
            total_recipes += 1
            
            # Batch upsert
            if len(vectors) >= batch_size:
                index.upsert(vectors)
                print(f"Upserted batch: {total_recipes} recipes processed")
                vectors = []
    
    # Upsert remaining vectors
    if vectors:
        index.upsert(vectors)
        print(f"Upserted final batch: {total_recipes} recipes total")
    
    return {
        "status": "success", 
        "message": f"Successfully ingested {total_recipes} recipes"
    }

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