import subprocess
import time
import sys
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import threading
import httpx

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
        response = await run_orchestrator(query_input.query)
        return QueryResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

def check_agent_health(port, agent_name):
    """Check if the agent server is running and responsive"""
    max_retries = 10
    for i in range(max_retries):
        try:
            with httpx.Client() as client:
                response = client.get(f"http://localhost:{port}/capabilities")
            if response.status_code == 200:
                print(f"{agent_name} is ready!")
                return True
        except httpx.RequestError:
            pass
        time.sleep(1)
    print(f"Failed to connect to {agent_name} on port {port} after {max_retries} retries.")
    return False

def start_agent_server(script_path, port, agent_name):
    """Start an agent server in a separate process and wait for it to be ready"""
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
    
    if not check_agent_health(port, agent_name):
        stdout, stderr = process.communicate()
        print(f"Failed to start {agent_name} on port {port}:")
        print(f"STDOUT: {stdout}")
        print(f"STDERR: {stderr}")
        process.terminate()
        return None
    
    print(f"Started {agent_name} on port {port}")
    return process

def start_orchestrator_api():
    """Start the orchestrator API server"""
    uvicorn.run("main:app", host="0.0.0.0", port=8008, reload=False)

if __name__ == "__main__":    # Start the agent servers
    research_agent_process = start_agent_server(
        "agents/research_agent.py", 
        8001,
        "Research Agent"
    )
    
    content_writing_process = start_agent_server(
        "agents/content_writing_agent.py", 
        8002,
        "Content Writing Agent"
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
    print("Orchestrator API running at: http://localhost:8008/query")
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
