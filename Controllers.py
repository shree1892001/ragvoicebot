import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from Services.RagBot import  OptimizedRAG
from typing import Tuple

app = FastAPI(
    title="Ultra-Fast RAG API",
    description="API wrapper for a blazing fast Retrieval-Augmented Generation bot.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, use specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


rag = OptimizedRAG()
rag.warm_up()



class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    response_time_ms: int


@app.get("/", tags=["Health"])
def health_check():
    return {"status": "RAG API is live 🚀"}


@app.post("/query", response_model=QueryResponse, tags=["Query"])
def ask_question(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    answer, response_time = rag.query(request.question)
    return QueryResponse(answer=answer, response_time_ms=response_time)


@app.post("/clear-cache", tags=["Admin"])
def clear_cache():
    rag._cached_query.cache_clear()
    return {"message": "Cache cleared successfully 🧹"}
def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()