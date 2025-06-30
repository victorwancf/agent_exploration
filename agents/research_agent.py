from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uvicorn
from langgraph.graph import StateGraph, END
import pandas as pd
import glob
import os

# Define the state for the research agent
default_data_dir = "../data"

class ResearchAgentState(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None
    result: Optional[str] = None
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

# The main logic node for the research agent
def research_node(state: ResearchAgentState) -> ResearchAgentState:
    query_text = state.query.lower()
    search_term = None
    # Airline-specific keywords
    if "skyglide" in query_text:
        search_term = "skyglide"
    elif "airvista" in query_text:
        search_term = "airvista"
    elif "aeroexpress" in query_text:
        search_term = "aeroexpress"
    elif "horizonhawk" in query_text:
        search_term = "horizonhawk"
    elif "flight delay" in query_text or "delayed" in query_text:
        search_term = "delay"
    elif "covid-19" in query_text or "covid" in query_text:
        search_term = "covid"
    elif "climate change" in query_text:
        search_term = "climate"
    elif "stock market" in query_text or "stock" in query_text:
        search_term = "stock"
    elif "self driving" in query_text or "self-driving" in query_text:
        search_term = "self driving"
    elif "metoo" in query_text:
        search_term = "metoo"
    # ... add more keyword logic as needed ...

    result = "No relevant data found."
    confidence = 0.5
    metadata = {}
    if search_term:
        files = glob.glob(os.path.join(default_data_dir, "*.csv"))
        for file in files:
            df = pd.read_csv(file)
            # Find rows where any cell contains the search_term
            mask = df.apply(lambda row: row.astype(str).str.lower().str.contains(search_term).any(), axis=1)
            matching_rows = df[mask]
            if not matching_rows.empty:
                # Try to extract feedback columns if present
                feedback_cols = [col for col in matching_rows.columns if 'feedback' in col.lower() or 'comment' in col.lower() or 'review' in col.lower()]
                if feedback_cols:
                    feedbacks = matching_rows[feedback_cols].astype(str).agg('\n'.join, axis=1).tolist()
                else:
                    feedbacks = matching_rows.astype(str).agg(', '.join, axis=1).tolist()
                # Limit to first 3 feedbacks for brevity
                feedback_sample = feedbacks[:3]
                feedback_text = '\n'.join(feedback_sample)
                result = f"Feedback for '{search_term}' from {os.path.basename(file)}:\n{feedback_text}"
                confidence = 0.95
                metadata = {"file": os.path.basename(file), "matches": len(matching_rows)}
                break
    else:
        result = "No specific search term found in query."
        confidence = 0.5
    return ResearchAgentState(
        query=state.query,
        context=state.context,
        result=result,
        confidence=confidence,
        metadata=metadata
    )

# Build the LangGraph for the research agent
workflow = StateGraph(ResearchAgentState)
workflow.add_node("research", research_node)
workflow.set_entry_point("research")
workflow.add_edge("research", END)
research_agent_app = workflow.compile()

# Example async runner for the agent
def run_research_agent(query: str, context: Optional[Dict[str, Any]] = None):
    state = ResearchAgentState(query=query, context=context)
    return research_agent_app.invoke(state)

app = FastAPI(title="Research Agent")

class Query(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None

class Response(BaseModel):
    result: str
    confidence: float
    metadata: Optional[Dict[str, Any]] = None

@app.post("/query", response_model=Response)
async def process_query(query_data: Query):
    # Convert Query to LangGraph state
    state = ResearchAgentState(query=query_data.query, context=query_data.context)
    result_state = research_agent_app.invoke(state)
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
        "capabilities": ["information_retrieval", "fact_checking", "literature_review", "source_discovery"],
        "name": "Research Agent",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    uvicorn.run("research_agent:app", host="0.0.0.0", port=8001, reload=True)
