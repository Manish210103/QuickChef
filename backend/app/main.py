# Updated main.py with organized Swagger tags and metadata

from fastapi import FastAPI, HTTPException, Query, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime
from bson import ObjectId
from collections import defaultdict
import numpy as np
from pymongo.errors import DuplicateKeyError

# Import your existing modules
from app.rag import ingest_data, query_rag, embed_text, get_pinecone_index
from app.auth import (
    connect_to_mongo, close_mongo_connection, get_database,
    hash_password, verify_password, create_access_token, verify_token
)

# Swagger metadata and tags

app = FastAPI(
    title="QuickChef RAG API",
    version="2.0.0",
    contact={
        "name": "QuickChef Support",
        "email": "support@quickchef.com",
    }
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect to MongoDB on startup
@app.on_event("startup")
def startup_db_client():
    connect_to_mongo()

@app.on_event("shutdown")
def shutdown_db_client():
    close_mongo_connection()

# Pydantic Models (same as before)
class UserRegister(BaseModel):
    username: str
    email: EmailStr
    password: str
    preferences: Optional[dict] = {}

class UserLogin(BaseModel):
    username: str
    password: str

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3
    cuisine_filter: Optional[str] = None
    max_time: Optional[int] = None

class HistoryAdd(BaseModel):
    recipe_id: str
    recipe_name: str
    rating: Optional[int] = None
    notes: Optional[str] = None

class UserPreferences(BaseModel):
    favorite_cuisines: Optional[List[str]] = []
    dietary_restrictions: Optional[List[str]] = []
    preferred_cooking_time: Optional[int] = None
    spice_level: Optional[str] = "medium"

# ===== SYSTEM ENDPOINTS =====
@app.get("/", 
         tags=["System"],
         )
def home():
    return {
        "message": "🍳 QuickChef RAG API v2.0",
        "features": [
            "🔍 Recipe Search with RAG",
            "👤 User Authentication",
            "📝 Cooking History Tracking", 
            "🎯 Personalized Recommendations",
            "📊 Cooking Analytics"
        ],
        "endpoints": {
            "auth": ["/auth/register", "/auth/login", "/auth/profile"],
            "search": ["/search", "/search/advanced"],
            "history": ["/history", "/history/add"],
            "recommendations": ["/recommendations"],
            "analytics": ["/analytics/cooking-stats"],
            "admin": ["/ingest", "/health"]
        }
    }

@app.get("/health",
         tags=["System"])
def health():
    """Health check endpoint"""
    try:
        # Test MongoDB connection
        db = get_database()
        db.command('ping')  # Simple ping command
        
        # Test Pinecone connection
        index = get_pinecone_index()
        stats = index.describe_index_stats()
        
        return {
            "status": "healthy",
            "mongodb": "connected",
            "pinecone": "connected",
            "vector_count": stats.total_vector_count if hasattr(stats, 'total_vector_count') else "unknown"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

# ===== AUTHENTICATION ENDPOINTS =====
@app.post("/auth/register",
          tags=["Authentication"])
def register_user(user: UserRegister):
    """Register a new user"""
    try:
        db = get_database()
        
        # Check if user exists
        existing_user = db.users.find_one({
            "$or": [{"username": user.username}, {"email": user.email}]
        })
        
        if existing_user is not None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username or email already registered"
            )
        
        # Create new user
        user_doc = {
            "username": user.username,
            "email": user.email,
            "hashed_password": hash_password(user.password),
            "preferences": user.preferences,
            "created_at": datetime.utcnow()
        }
        
        try:
            result = db.users.insert_one(user_doc)
        except DuplicateKeyError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username or email already registered"
            )
        
        # Create access token
        access_token = create_access_token(
            data={"sub": user.username, "user_id": str(result.inserted_id)}
        )
        
        return {
            "message": "User registered successfully",
            "access_token": access_token,
            "token_type": "bearer",
            "user_id": str(result.inserted_id)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration error: {str(e)}")

@app.post("/auth/login",
          tags=["Authentication"])
def login_user(user: UserLogin):
    """Login user and return JWT token"""
    try:
        db = get_database()
        
        # Find user
        user_doc = db.users.find_one({"username": user.username})
        
        if user_doc is None or not verify_password(user.password, user_doc["hashed_password"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )
        
        # Create access token
        access_token = create_access_token(
            data={"sub": user.username, "user_id": str(user_doc["_id"])}
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user_id": str(user_doc["_id"]),
            "username": user.username
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Login error: {str(e)}")

@app.get("/auth/profile",
         tags=["Authentication"])
def get_user_profile(current_user: dict = Depends(verify_token)):
    """Get current user profile"""
    try:
        db = get_database()
        user = db.users.find_one({"_id": ObjectId(current_user["user_id"])})
        
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {
            "user_id": str(user["_id"]),
            "username": user["username"],
            "email": user["email"],
            "preferences": user.get("preferences", {}),
            "created_at": user["created_at"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profile error: {str(e)}")

@app.put("/auth/preferences",
         tags=["Authentication"])
def update_preferences(
    preferences: UserPreferences,
    current_user: dict = Depends(verify_token)
):
    """Update user preferences"""
    try:
        db = get_database()
        
        result = db.users.update_one(
            {"_id": ObjectId(current_user["user_id"])},
            {"$set": {"preferences": preferences.dict()}}
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {"message": "Preferences updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preferences update error: {str(e)}")

# ===== RECIPE SEARCH ENDPOINTS =====
@app.get("/search",
         tags=["Recipe Search"])
def search(
    query: str = Query(..., description="Search query (e.g., 'spicy chicken curry')", example="butter chicken curry"),
    top_k: int = Query(3, ge=1, le=10, description="Number of results to return")
):
    """Public recipe search"""
    try:
        results = query_rag(query, top_k)
        return {"query": query, "results": results, "total": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/advanced",
          tags=["Recipe Search"])
def search_advanced(request: QueryRequest):
    """Advanced recipe search with filters"""
    try:
        filters = {}
        if request.cuisine_filter:
            filters["cuisine"] = {"$eq": request.cuisine_filter}
        if request.max_time:
            filters["total_time"] = {"$lte": request.max_time}
        
        results = query_rag(request.query, request.top_k, filters if filters else None)
        return {"query": request.query, "results": results, "total": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ===== COOKING HISTORY ENDPOINTS =====
@app.post("/history/add",
          tags=["Cooking History"])
def add_to_history(
    history_item: HistoryAdd,
    current_user: dict = Depends(verify_token)
):
    """Add recipe to user's cooking history"""
    try:
        db = get_database()
        
        history_doc = {
            "user_id": ObjectId(current_user["user_id"]),
            "recipe_id": history_item.recipe_id,
            "recipe_name": history_item.recipe_name,
            "rating": history_item.rating,
            "notes": history_item.notes,
            "cooked_at": datetime.utcnow()
        }
        
        result = db.user_history.insert_one(history_doc)
        
        return {
            "message": "Recipe added to history",
            "history_id": str(result.inserted_id)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"History add error: {str(e)}")

@app.get("/history",
         tags=["Cooking History"])
def get_user_history(
    limit: int = Query(20, ge=1, le=100, description="Maximum number of history items to return"),
    skip: int = Query(0, ge=0, description="Number of items to skip for pagination"),
    current_user: dict = Depends(verify_token)
):
    """Get user's cooking history"""
    try:
        db = get_database()
        
        cursor = db.user_history.find(
            {"user_id": ObjectId(current_user["user_id"])}
        ).sort("cooked_at", -1).skip(skip).limit(limit)
        
        history = []
        for doc in cursor:
            doc["_id"] = str(doc["_id"])
            doc["user_id"] = str(doc["user_id"])
            history.append(doc)
        
        total_count = db.user_history.count_documents(
            {"user_id": ObjectId(current_user["user_id"])}
        )
        
        return {
            "history": history,
            "total": total_count,
            "limit": limit,
            "skip": skip
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"History retrieval error: {str(e)}")

@app.delete("/history/{history_id}",
            tags=["Cooking History"])
def delete_history_item(
    history_id: str,
    current_user: dict = Depends(verify_token)
):
    """Delete a history item"""
    try:
        db = get_database()
        
        result = db.user_history.delete_one({
            "_id": ObjectId(history_id),
            "user_id": ObjectId(current_user["user_id"])
        })
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="History item not found")
        
        return {"message": "History item deleted successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"History delete error: {str(e)}")

# ===== RECOMMENDATIONS ENDPOINT =====
@app.get("/recommendations",
         tags=["Recommendations"])
def get_recommendations(
    count: int = Query(5, ge=1, le=20, description="Number of recommendations to return"),
    current_user: dict = Depends(verify_token)
):
    """Get personalized recipe recommendations based on user history and preferences"""
    try:
        db = get_database()
        
        # Get user preferences
        user = db.users.find_one({"_id": ObjectId(current_user["user_id"])})
        preferences = user.get("preferences", {}) if user else {}
        
        # Get user history
        history_cursor = db.user_history.find(
            {"user_id": ObjectId(current_user["user_id"])}
        ).sort("cooked_at", -1).limit(20)
        
        history_recipes = [doc["recipe_name"] for doc in history_cursor]
        
        # Build recommendation query based on preferences and history
        query_parts = []
        
        # Add favorite cuisines
        favorite_cuisines = preferences.get("favorite_cuisines", [])
        if favorite_cuisines:
            query_parts.extend(favorite_cuisines)
        
        # Add dietary restrictions context
        dietary_restrictions = preferences.get("dietary_restrictions", [])
        if dietary_restrictions:
            query_parts.extend([f"no {restriction}" for restriction in dietary_restrictions])
        
        # Add spice level preference
        spice_level = preferences.get("spice_level", "medium")
        query_parts.append(f"{spice_level} spice")
        
        # Create query from history (find similar recipes)
        if history_recipes:
            query_parts.extend(history_recipes[:3])  # Use last 3 recipes
        
        # Build final query
        query = " ".join(query_parts[:10])  # Limit query length
        if not query:
            query = "popular Indian recipes"
        
        # Build filters
        filters = {}
        if favorite_cuisines:
            filters["cuisine"] = {"$in": favorite_cuisines}
        
        preferred_time = preferences.get("preferred_cooking_time")
        if preferred_time:
            filters["total_time"] = {"$lte": preferred_time}
        
        # Get recommendations
        results = query_rag(
            query=query,
            top_k=count * 2,  # Get more to filter duplicates
            filters=filters if filters else None
        )
        
        # Filter out recipes already in history
        history_recipe_ids = set()
        history_cursor_2 = db.user_history.find(
            {"user_id": ObjectId(current_user["user_id"])}
        )
        for doc in history_cursor_2:
            history_recipe_ids.add(doc["recipe_id"])
        
        filtered_results = []
        for result in results:
            # Extract recipe_id from the result (assuming it's in the format "recipe_X")
            recipe_score = result["score"]
            if recipe_score > 0.3:  # Only include relevant matches
                filtered_results.append(result)
            
            if len(filtered_results) >= count:
                break
        
        return {
            "recommendations": filtered_results[:count],
            "query_used": query,
            "user_preferences": preferences,
            "total_found": len(filtered_results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ===== ANALYTICS ENDPOINT =====
@app.get("/analytics/cooking-stats",
         tags=["Analytics"])
def get_cooking_stats(current_user: dict = Depends(verify_token)):
    """Get user's cooking analytics"""
    try:
        db = get_database()
        
        # Total recipes cooked
        total_cooked = db.user_history.count_documents(
            {"user_id": ObjectId(current_user["user_id"])}
        )
        
        # Get recent activity (last 30 days)
        from datetime import datetime, timedelta
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        
        recent_activity = db.user_history.count_documents({
            "user_id": ObjectId(current_user["user_id"]),
            "cooked_at": {"$gte": thirty_days_ago}
        })
        
        # Average rating
        rating_pipeline = [
            {"$match": {
                "user_id": ObjectId(current_user["user_id"]),
                "rating": {"$exists": True, "$ne": None}
            }},
            {"$group": {
                "_id": None,
                "avg_rating": {"$avg": "$rating"},
                "total_rated": {"$sum": 1}
            }}
        ]
        
        rating_result = list(db.user_history.aggregate(rating_pipeline))
        avg_rating = rating_result[0]["avg_rating"] if rating_result else None
        total_rated = rating_result[0]["total_rated"] if rating_result else 0
        
        return {
            "total_recipes_cooked": total_cooked,
            "recent_activity_30_days": recent_activity,
            "average_rating": round(avg_rating, 2) if avg_rating else None,
            "total_rated_recipes": total_rated,
            "cooking_streak": "Feature coming soon"  # You can implement streak logic
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ===== ADMIN ENDPOINTS =====
@app.post("/ingest",
          tags=["Admin"],
          description="Load recipe dataset into Pinecone vector database (Admin operation)")
def ingest():
    """Ingest recipe dataset into Pinecone (Admin only - add auth if needed)"""
    try:
        result = ingest_data()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))