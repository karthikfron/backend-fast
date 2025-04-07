from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import numpy as np
import json
import google.generativeai as genai

app = FastAPI()

# Load FAISS index
index = faiss.read_index("faiss_index.index")

# Load assessment data
with open("enriched_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Request model
class Query(BaseModel):
    text: str

# Gemini embedding generator
def generate_embedding(text):
    try:
        response = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_query"
        )
        return response['embedding']
    except Exception as e:
        print(f"Error: {e}")
        return None

# FAISS search
def search_assessments(query_embedding, top_k=10):
    distances, indices = index.search(np.array([query_embedding]).astype("float32"), top_k)
    return [data[i] for i in indices[0]]

#  FIXED API Endpoint
@app.post("/search")
def search(query: Query):  
    embedding = generate_embedding(query.text)
    if not embedding:
        return {"error": "Failed to generate embedding"}
    # Normalize the embedding
    
    results = search_assessments(embedding)
    return {
        "results": [
            {
                "title": r.get("title", "N/A"),
                "duration": r.get("duration", "N/A"),
                "type": r.get("type", "N/A"),
                "remote": r.get("remote", "N/A"),
                "adaptive": r.get("adaptive", "N/A"),
                "url": r.get("url", "No link available")
            }
            for r in results
        ]
    }
