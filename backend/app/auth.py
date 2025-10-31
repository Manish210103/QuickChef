# auth.py - Authentication module using Argon2 with SHA256 pre-hashing for long passwords
import os
from datetime import datetime, timedelta
import hashlib
from passlib.context import CryptContext
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError, ConnectionFailure
from jose import JWTError, jwt

# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-this-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 24 hours

# Use Argon2 for password hashing - no character length limitations
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")
security = HTTPBearer()

# MongoDB Configuration
MONGODB_URL = os.getenv("MONGODB_URL")
DATABASE_NAME = "quickchef"

# Global MongoDB client and database
mongo_client = None
mongo_db = None

def connect_to_mongo():
    """Create database connection"""
    global mongo_client, mongo_db
    
    if not MONGODB_URL:
        raise HTTPException(
            status_code=500, 
            detail="MONGODB_URL environment variable not set"
        )
    
    try:
        mongo_client = MongoClient(MONGODB_URL, serverSelectionTimeoutMS=5000)
        # Test the connection
        mongo_client.admin.command('ismaster')
        mongo_db = mongo_client[DATABASE_NAME]
        
        # Create indexes for better performance
        try:
            mongo_db.users.create_index("username", unique=True)
            mongo_db.users.create_index("email", unique=True)
            mongo_db.user_history.create_index([("user_id", 1), ("cooked_at", -1)])
        except Exception as e:
            print(f"Warning: Could not create indexes: {e}")
            
    except ConnectionFailure as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not connect to MongoDB: {e}"
        )

def close_mongo_connection():
    """Close database connection"""
    global mongo_client
    if mongo_client is not None:
        mongo_client.close()

def get_database():
    """Get database instance"""
    global mongo_db
    if mongo_db is None:
        connect_to_mongo()
    return mongo_db

def _normalize_password(password: str) -> str:
    """Normalize password: if longer than 72 bytes, SHA256 hash it first
    
    This allows Argon2 to handle passwords of any length by converting
    long passwords to fixed-length SHA256 hashes before argon2 hashing.
    """
    password_bytes = password.encode('utf-8')
    
    # If password is longer than 72 bytes, hash it with SHA256 first
    if len(password_bytes) > 72:
        # Use SHA256 to normalize to fixed length, then convert to hex string
        sha256_hash = hashlib.sha256(password_bytes).hexdigest()
        return sha256_hash
    
    return password

def hash_password(password: str) -> str:
    """Hash a password using Argon2 with SHA256 pre-hashing for long passwords
    
    Supports passwords of any length:
    - Short passwords (<=72 bytes): hashed directly with Argon2
    - Long passwords (>72 bytes): SHA256 hashed first, then Argon2 hashed
    
    This gives us the security benefits of Argon2 without length limitations.
    """
    normalized = _normalize_password(password)
    return pwd_context.hash(normalized)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against Argon2 hash
    
    Uses the same normalization logic as hash_password to ensure
    consistency for both short and long passwords.
    """
    try:
        normalized = _normalize_password(plain_password)
        return pwd_context.verify(normalized, hashed_password)
    except Exception as e:
        print(f"Password verification error: {e}")
        return False

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token and return user info"""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        user_id: str = payload.get("user_id")
        
        if username is None or user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        return {"username": username, "user_id": user_id}
    
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired"
        )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )