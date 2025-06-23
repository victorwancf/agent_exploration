import subprocess
import time
import sys
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import threading

from orchestrator.orchestrator_agent import run_orchestrator

class QueryInput(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str

app = FastAPI(title="Agent Orchestrator API")

@app.post("/query")
async def process_query(query_input: QueryInput):
    """Process a query through the orchestrator agent"""
    try:
        response = run_orchestrator(query_input.query)
        return QueryResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

def start_agent_server(script_path, port):
    """Start an agent server in a separate process"""
    # Get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(script_path))
    script_name = os.path.basename(script_path)
    
    # Change to the script's directory and run the server
    # This ensures that imports in the script will work correctly
    cmd = [sys.executable, script_name]
    
    # Start the process
    process = subprocess.Popen(
        cmd,
        cwd=script_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for the server to start
    time.sleep(2)
    
    # Check if the process is still running
    if process.poll() is not None:
        stdout, stderr = process.communicate()
        print(f"Failed to start agent on port {port}:")
        print(f"STDOUT: {stdout}")
        print(f"STDERR: {stderr}")
        return None
    
    print(f"Started agent server on port {port}")
    return process

def start_orchestrator_api():
    """Start the orchestrator API server"""
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":    # Start the agent servers
    research_agent_process = start_agent_server(
        "agents/research_agent.py", 
        8001
    )
    
    content_writing_process = start_agent_server(
        "agents/content_writing_agent.py", 
        8002
    )
    
    # Check if both servers started successfully
    if research_agent_process is None or content_writing_process is None:
        print("Failed to start one or more agent servers. Exiting.")
        if research_agent_process is not None:
            research_agent_process.terminate()
        if content_writing_process is not None:
            content_writing_process.terminate()
        sys.exit(1)
      # Start the orchestrator API in a separate thread
    orchestrator_thread = threading.Thread(target=start_orchestrator_api, daemon=True)
    orchestrator_thread.start()
    
    print("All services started!")
    print("Orchestrator API running at: http://localhost:8000/query")
    print("Research Agent running at: http://localhost:8001/query")
    print("Content Writing Agent running at: http://localhost:8002/query")
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        # Terminate the processes
        research_agent_process.terminate()
        content_writing_process.terminate()
        # Wait for processes to end
        research_agent_process.wait()
        content_writing_process.wait()
        print("All services stopped.")
