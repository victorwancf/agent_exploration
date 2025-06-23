from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uvicorn

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

    # Simulate writing responses based on the query
    if "summarize" in query_text or "summary" in query_text:
        return Response(
            result="Here is a concise summary of the provided content: The main points are clearly outlined, and the overall message is preserved.",
            confidence=0.93,
            metadata={"task": "summarization", "length": "short"}
        )
    elif "edit" in query_text or "improve" in query_text or "grammar" in query_text:
        return Response(
            result="The text has been edited for clarity, grammar, and style. The revised version is more readable and engaging.",
            confidence=0.91,
            metadata={"task": "editing", "improvements": ["clarity", "grammar", "style"]}
        )
    elif "write" in query_text or "create" in query_text or "generate" in query_text:
        return Response(
            result="Here is the requested content: This article introduces the topic, provides key insights, and concludes with actionable recommendations.",
            confidence=0.90,
            metadata={"task": "content_creation", "format": "article"}
        )
    elif "adapt" in query_text or "style" in query_text:
        return Response(
            result="The content has been adapted to match the requested style. Tone and vocabulary have been adjusted accordingly.",
            confidence=0.88,
            metadata={"task": "style_adaptation", "style": "requested"}
        )
    else:
        return Response(
            result="General content writing assistance provided. Please specify if you need writing, editing, summarization, or style adaptation.",
            confidence=0.75,
            metadata={"general_writing": True}
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