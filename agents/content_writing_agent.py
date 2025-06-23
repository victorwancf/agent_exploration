from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uvicorn
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

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
    query_text = query_data.query.lower()
    model = genai.GenerativeModel('gemini-1.5-flash')

    task = "general"
    prompt = f"Fulfill the following request: '{query_data.query}'"

    if any(k in query_text for k in ["edit", "improve", "grammar", "clarity", "fix"]):
        task = "editing"
        prompt = f"The user wants to edit a piece of text. Here is their request: '{query_data.query}'. Please provide only the edited text as a response."
    elif any(k in query_text for k in ["write", "create", "generate", "draft"]):
        task = "content_creation"
        prompt = f"The user wants to create content. Here is their request: '{query_data.query}'. Please provide only the generated content as a response."
    elif any(k in query_text for k in ["summarize", "summary", "tl;dr", "shorten"]):
        task = "summarization"
        prompt = f"The user wants to summarize a piece of text. Here is their request: '{query_data.query}'. Please provide only the summary as a response."
    elif any(k in query_text for k in ["adapt", "style", "tone", "rewrite"]):
        task = "style_adaptation"
        prompt = f"The user wants to adapt the style of a piece of text. Here is their request: '{query_data.query}'. Please provide only the adapted text as a response."

    response = model.generate_content(prompt)
    response_text = response.text if hasattr(response, 'text') else response.candidates[0].content.parts[0].text

    return Response(
        result=response_text,
        confidence=0.95,
        metadata={"task": task}
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