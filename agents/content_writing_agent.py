from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uvicorn
import google.generativeai as genai
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
import logging

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# Define the state for the content writing agent
class ContentWritingAgentState(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None
    result: Optional[str] = None
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

# The main logic node for the content writing agent
def content_writing_node(state: ContentWritingAgentState) -> ContentWritingAgentState:
    try:
        query_text = state.query.lower()
        
        if not GEMINI_API_KEY:
            error_msg = "GOOGLE_API_KEY is not set. Please check your .env file."
            logging.error(error_msg)
            raise ValueError(error_msg)

        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        task = "general"
        # More specific prompt engineering
        if any(k in query_text for k in ["edit", "improve", "grammar", "clarity", "fix"]):
            task = "editing"
            prompt = f"Please edit the following text for grammar and clarity, providing only the improved text as your response: '{state.query}'"
        elif any(k in query_text for k in ["write", "create", "generate", "draft"]):
            task = "content_creation"
            prompt = f"Please create content based on the following request, providing only the generated content as your response: '{state.query}'"
        elif any(k in query_text for k in ["summarize", "summary", "tl;dr", "shorten"]):
            task = "summarization"
            prompt = f"Please summarize the following text, providing only the summary as your response: '{state.query}'"
        elif any(k in query_text for k in ["adapt", "style", "tone", "rewrite"]):
            task = "style_adaptation"
            prompt = f"Please adapt the style or tone of the following text based on the request, providing only the adapted text as your response: '{state.query}'"
        else:
            prompt = f"Fulfill the following request: '{state.query}'"

        logging.info(f"Sending prompt to Gemini for task '{task}': {prompt}")
        response = model.generate_content(prompt)
        
        if not response.candidates:
            error_message = "The response from the model was empty or blocked, possibly due to safety settings."
            logging.error(error_message)
            return ContentWritingAgentState(
                query=state.query,
                context=state.context,
                result=error_message,
                confidence=0.5,
                metadata={"task": task, "error": "Blocked or empty response"}
            )

        response_text = response.text
        
        return ContentWritingAgentState(
            query=state.query,
            context=state.context,
            result=response_text,
            confidence=0.95,
            metadata={"task": task}
        )
    except Exception as e:
        logging.error(f"An error occurred in content_writing_node: {e}", exc_info=True)
        return ContentWritingAgentState(
            query=state.query,
            context=state.context,
            result=f"An internal error occurred in the Content Writing Agent: {str(e)}",
            confidence=0.1,
            metadata={"task": "error"}
        )

# Build the LangGraph for the content writing agent
workflow = StateGraph(ContentWritingAgentState)
workflow.add_node("content_writer", content_writing_node)
workflow.set_entry_point("content_writer")
workflow.add_edge("content_writer", END)
content_writing_agent_app = workflow.compile()

app = FastAPI(title="Content Writing Agent")

class Query(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None

class Response(BaseModel):
    result: str
    confidence: float
    metadata: Optional[Dict[str, Any]] = None

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
    # Convert Query to LangGraph state
    state = ContentWritingAgentState(query=query_data.query, context=query_data.context)
    result_state = content_writing_agent_app.invoke(state)
    
    # If result_state is a dict, use keys; if it's an object, use attributes
    if isinstance(result_state, dict):
        return Response(
            result=str(result_state.get("result") or ""),
            confidence=result_state.get("confidence", 0.0),
            metadata=result_state.get("metadata")
        )
    else:
        return Response(
            result=str(getattr(result_state, "result", "") or ""),
            confidence=getattr(result_state, "confidence", 0.0),
            metadata=getattr(result_state, "metadata", None)
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