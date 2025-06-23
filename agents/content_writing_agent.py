from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uvicorn
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

# Set up Gemini API key
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

class Query(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None

class Response(BaseModel):
    result: str
    confidence: float
    metadata: Optional[Dict[str, Any]] = None

app = FastAPI(title="Content Writing Agent")

@app.post("/query", response_model=Response)
async def process_query(query_data: Query):
    """
    Process a content writing-related query and return generated content.
    
    This agent specializes in:
    - Content creation
    - Editing
    - Style adaptation
    - Summarization
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"Please edit the following sentence for grammar and clarity: {query_data.query}"
    response = model.generate_content(prompt)
    response_text = response.text if hasattr(response, 'text') else response.candidates[0].content.parts[0].text

    return Response(
        result=response_text,
        confidence=0.95,  # You might want to derive confidence from the model's response
        metadata={"task": "editing"}
    )

@app.get("/capabilities")
async def get_capabilities():
    """Return the capabilities of this agent"""
    return {
        "capabilities": ["content_creation", "editing", "style_adaptation", "summarization"],
        "name": "Content Writing Agent",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    uvicorn.run("content_writing_agent:app", host="0.0.0.0", port=8002, reload=True)