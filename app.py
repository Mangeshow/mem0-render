import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mem0 import Memory
from typing import Optional, List, Dict, Any

app = FastAPI(title="Mem0 API", version="1.0.0")

# Enklare konfiguration för mem0 (använder standardvärden)
# Låt mem0 hantera konfigurationen automatiskt
try:
    memory = Memory()
except Exception as e:
    print(f"Error initializing Memory with default config: {e}")
    # Fallback till minimal konfiguration
    memory = Memory({
        "vector_store": {
            "provider": "chroma",
            "config": {
                "path": "/app/data/chroma_db"
            }
        }
    })

class AddMemoryRequest(BaseModel):
    text: str
    user_id: Optional[str] = "default_user"
    metadata: Optional[Dict[str, Any]] = None

class SearchMemoryRequest(BaseModel):
    query: str
    user_id: Optional[str] = "default_user"
    limit: Optional[int] = 10

class UpdateMemoryRequest(BaseModel):
    memory_id: str
    text: str
    user_id: Optional[str] = "default_user"

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Mem0 API is running!",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/v1/memories")
async def add_memory(request: AddMemoryRequest):
    try:
        # Använd korrekt API för mem0
        result = memory.add(
            request.text,
            user_id=request.user_id,
            metadata=request.metadata or {}
        )
        return {
            "success": True,
            "memory_id": result.get("id") if isinstance(result, dict) else str(result),
            "message": "Memory added successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding memory: {str(e)}")

@app.get("/v1/memories")
async def get_memories(user_id: str = "default_user"):
    try:
        memories = memory.get_all(user_id=user_id)
        return {
            "success": True,
            "memories": memories,
            "count": len(memories) if memories else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting memories: {str(e)}")

@app.post("/v1/memories/search")
async def search_memories(request: SearchMemoryRequest):
    try:
        results = memory.search(
            query=request.query,
            user_id=request.user_id,
            limit=request.limit
        )
        return {
            "success": True,
            "results": results,
            "count": len(results) if results else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching memories: {str(e)}")

@app.put("/v1/memories/{memory_id}")
async def update_memory(memory_id: str, request: UpdateMemoryRequest):
    try:
        result = memory.update(
            memory_id=memory_id,
            data=request.text,
            user_id=request.user_id
        )
        return {
            "success": True,
            "message": "Memory updated successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating memory: {str(e)}")

@app.delete("/v1/memories/{memory_id}")
async def delete_memory(memory_id: str, user_id: str = "default_user"):
    try:
        memory.delete(memory_id=memory_id, user_id=user_id)
        return {
            "success": True,
            "message": "Memory deleted successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting memory: {str(e)}")

@app.delete("/v1/memories")
async def delete_all_memories(user_id: str = "default_user"):
    try:
        memory.delete_all(user_id=user_id)
        return {
            "success": True,
            "message": "All memories deleted successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting all memories: {str(e)}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
