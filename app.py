import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mem0 import Memory
from typing import Optional, List, Dict, Any

app = FastAPI(title="Mem0 API", version="1.0.0")

config = {
    "vector_store": {
        "provider": "chroma",
        "config": {
            "path": "/app/data/chroma_db"
        }
    },
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4o-mini",
            "temperature": 0.2,
            "max_tokens": 1500,
        }
    }
}

memory = Memory(config=config)

class AddMemoryRequest(BaseModel):
    text: str
    user_id: Optional[str] = "default_user"
    metadata: Optional[Dict[str, Any]] = None

@app.get("/")
async def root():
    return {
        "message": "Mem0 API is running!",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.post("/v1/memories")
async def add_memory(request: AddMemoryRequest):
    try:
        result = memory.add(
            messages=request.text,
            user_id=request.user_id,
            metadata=request.metadata or {}
        )
        return {
            "success": True,
            "memory_id": result.get("id"),
            "message": "Memory added successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
