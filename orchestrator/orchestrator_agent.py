from typing import Dict, List, Any, Optional, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
import configparser
import httpx
import os
import google.generativeai as genai
import asyncio
import json
from langgraph.graph import StateGraph, END

# Define the state for our graph
class AgentState(TypedDict):
    messages: List[BaseMessage]
    current_agent: Optional[str]
    next_action: Literal["ROUTE", "PROCESS", "END"]
    response: Optional[str]

# Function to read agent registry
def read_agent_registry(file_path: str) -> Dict[str, Dict[str, Any]]:
    """Read agent registry file and return parsed data."""
    config = configparser.ConfigParser()
    config.read(file_path)
    
    registry = {}
    for section in config.sections():
        registry[section] = {
            "name": config.get(section, "name"),
            "description": config.get(section, "description"),
            "endpoint": config.get(section, "endpoint"),
            "capabilities": json.loads(config.get(section, "capabilities").replace("'", '"'))
        }
    
    return registry

# Initialize the agent registry
agent_registry = read_agent_registry("config/agent_registry.txt")

# Set up Gemini API key
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Define a router model that will analyze the query and determine which agent to use
def route_query(state: AgentState) -> AgentState:
    """Route the query to the appropriate agent based on its content."""
    # Extract the last human message
    human_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    if not human_messages:
        return {**state, "next_action": "END", "response": "No query provided."}
    
    query = human_messages[-1].content
    
    # Use a template to create a router prompt
    router_template = """You are an orchestrator agent that routes queries to specialized agents.\n\nAvailable agents:\n{agent_descriptions}\n\nUser query: {query}\n\nAnalyze the query and determine which agent is best suited to handle it.\nSelect exactly one agent from the list above.\n\nOutput format: \nAGENT_NAME: <selected agent>\nREASON: <brief explanation for selection>\n"""
    
    # Prepare agent descriptions
    agent_descriptions = ""
    for agent_id, details in agent_registry.items():
        capabilities = ", ".join(details["capabilities"])
        agent_descriptions += f"- {agent_id}: {details['name']} - {details['description']} (Capabilities: {capabilities})\n"
    
    router_prompt = router_template.format(agent_descriptions=agent_descriptions, query=query)
    
    # Use Gemini to determine the routing
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(router_prompt)
    response_text = response.text if hasattr(response, 'text') else response.candidates[0].content.parts[0].text
    
    # Parse the response to extract the selected agent
    selected_agent = None
    for line in response_text.split("\n"):
        if line.startswith("AGENT_NAME:"):
            selected_agent = line.split(":", 1)[1].strip()
            break
    
    # If we couldn't determine an agent, provide a helpful response
    if not selected_agent or selected_agent not in agent_registry:
        return {
            **state, 
            "next_action": "END",
            "response": "I couldn't determine which agent would be best suited for your query. Could you please provide more specific information?"
        }
    
    return {
        **state,
        "current_agent": selected_agent,
        "next_action": "PROCESS"
    }

# Define a function to process the query with the selected agent
async def process_with_agent(state: AgentState) -> AgentState:
    """Process the query using the selected agent."""
    agent_id = state["current_agent"]
    if not agent_id or agent_id not in agent_registry:
        return {
            **state,
            "next_action": "END",
            "response": "No valid agent was selected for this query."
        }
    
    agent_details = agent_registry[agent_id]
    endpoint = agent_details["endpoint"]
    
    # Extract the query from the last human message
    human_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    if not human_messages:
        return {**state, "next_action": "END", "response": "No query provided."}
    
    query = human_messages[-1].content
    
    # Send the query to the agent's API
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                endpoint,
                json={"query": query},
                timeout=10.0
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Format the response
                formatted_response = (
                    f"Response from {agent_details['name']}:\n\n"
                    f"{result['result']}\n\n"
                    f"Confidence: {result['confidence']}"
                )
                
                if result.get('metadata'):
                    metadata_str = "\n".join([f"- {k}: {v}" for k, v in result['metadata'].items()])
                    formatted_response += f"\n\nAdditional Information:\n{metadata_str}"
                
                return {
                    **state,
                    "next_action": "END",
                    "response": formatted_response
                }
            else:
                return {
                    **state,
                    "next_action": "END",
                    "response": f"Error from {agent_details['name']}: {response.text}"
                }
        except Exception as e:
            return {
                **state,
                "next_action": "END",
                "response": f"Failed to communicate with {agent_details['name']}: {str(e)}"
            }

# Helper function to handle the asynchronous process_with_agent
def process_with_agent_sync(state: AgentState) -> AgentState:
    """Synchronous wrapper for the asynchronous process_with_agent function."""
    return asyncio.run(process_with_agent(state))

# Define a function to decide the next step in the workflow
def decide_next_step(state: AgentState) -> Literal["route", "process", "end"]:
    """Determine the next step based on the state."""
    return state["next_action"].lower()

# Build the graph
def build_orchestrator_graph():
    """Build and return the orchestrator graph."""
    workflow = StateGraph(AgentState)
    
    # Add the nodes
    workflow.add_node("route", route_query)
    workflow.add_node("process", process_with_agent_sync)
    
    # Define the conditional edges
    workflow.add_conditional_edges(
        "route",
        {
            "process": lambda state: state["next_action"] == "PROCESS",
            END: lambda state: state["next_action"] == "END",
        }
    )
    workflow.add_edge("process", END)
    
    # Set the entry point
    workflow.set_entry_point("route")
    
    return workflow.compile()

# Create the graph
orchestrator_graph = build_orchestrator_graph()

# Define a function to run the orchestrator
def run_orchestrator(query: str) -> str:
    """Run the orchestrator with a given query and return the response."""
    # Initialize the state
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "current_agent": None,
        "next_action": "ROUTE",
        "response": None
    }
    
    # Run the graph
    final_state = orchestrator_graph.invoke(initial_state)
    
    # Return the response
    return final_state["response"]

if __name__ == "__main__":
    # Example usage
    sample_query = "Can you analyze this dataset and tell me what patterns you see?"
    result = run_orchestrator(sample_query)
    print(result)
