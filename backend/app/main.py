# backend/app/main.py

from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(title="8B RAG Backend", version="0.1.0")


@app.get("/health", tags=["system"])
def health_check():
    return JSONResponse({"status": "ok", "message": "Hello world from 8B RAG backend!"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
