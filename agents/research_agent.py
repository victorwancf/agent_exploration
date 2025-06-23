from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uvicorn
import pandas as pd
import glob

class Query(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None

class Response(BaseModel):
    result: str
    confidence: float
    metadata: Optional[Dict[str, Any]] = None

app = FastAPI(title="Research Agent")

DATA_DIR = "data"

@app.post("/query", response_model=Response)
async def process_query(query_data: Query):
    """
    Process a research-related query and return findings.
    Now supports searching and analyzing Twitter post CSVs in the data directory.
    
    This agent specializes in:
    - Information retrieval
    - Fact checking
    - Literature review
    - Source discovery
    """
    query_text = query_data.query.lower()
    
    # If the query is about searching, finding, or analyzing, search the CSVs
    if any(k in query_text for k in ["find", "search", "analyze", "information", "review", "posts", "tweet", "twitter"]):
        # Gather all CSV files in the data directory
        csv_files = glob.glob(f"{DATA_DIR}/*.csv")
        results = []
        for file in csv_files:
            try:
                df = pd.read_csv(file, header=None, names=["post"])
                # Search for posts containing any keyword from the query
                matches = df[df["post"].str.lower().str.contains(query_text, na=False)]
                if not matches.empty:
                    results.append({
                        "file": file.split("/")[-1],
                        "count": len(matches),
                        "examples": matches["post"].head(3).tolist()
                    })
            except Exception as e:
                continue
        if results:
            summary = f"Found relevant posts in {len(results)} file(s).\n"
            for r in results:
                summary += f"\nFile: {r['file']} (Matches: {r['count']})\nExamples: " + " | ".join(r['examples'])
            return Response(
                result=summary,
                confidence=0.95,
                metadata={"files_with_matches": [r["file"] for r in results], "total_matches": sum(r["count"] for r in results)}
            )
        else:
            return Response(
                result="No relevant posts found in the Twitter data.",
                confidence=0.5,
                metadata={"searched_files": [f.split("/")[-1] for f in csv_files]}
            )
    # Simulate research responses based on the query
    elif "find" in query_text or "search" in query_text or "information" in query_text:
        return Response(
            result="I've found several relevant sources on this topic. The most recent research from Stanford (2024) indicates that the hypothesis has strong empirical support across multiple studies.",
            confidence=0.89,
            metadata={"sources": 12, "primary_sources": 7, "recency": "high", "consensus_level": "strong"}
        )
    elif "fact check" in query_text or "verify" in query_text or "validate" in query_text:
        return Response(
            result="This claim appears to be partially accurate but missing important context. While the core statement is supported by evidence, there are significant qualifications noted in the literature.",
            confidence=0.92,
            metadata={"accuracy_rating": "partially accurate", "primary_sources_checked": 5, "contradictory_evidence": "minimal"}
        )
    elif "literature" in query_text or "review" in query_text or "papers" in query_text:
        return Response(
            result="The literature review reveals three major schools of thought on this topic. The dominant view (supported by 65% of recent papers) favors the mechanistic explanation, while competing theories focus on emergent properties and contextual factors.",
            confidence=0.94,
            metadata={"papers_reviewed": 47, "time_period": "2020-2025", "major_researchers": ["Zhang", "Patel", "Yamamoto"]}
        )
    elif "background" in query_text or "context" in query_text:
        return Response(
            result="This field emerged in the early 2010s and has seen exponential growth since 2018. The foundational work by Rodriguez et al. established the theoretical framework that most current research builds upon.",
            confidence=0.91,
            metadata={"historical_depth": "comprehensive", "key_developments": 4, "paradigm_shifts": 1}
        )
    else:
        return Response(
            result="I've researched your query and compiled relevant information from authoritative sources. The consensus view suggests the phenomenon is well-established, though some aspects remain under investigation.",
            confidence=0.80,
            metadata={"general_research": True, "confidence_factors": ["topic breadth", "evolving field"]}
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
