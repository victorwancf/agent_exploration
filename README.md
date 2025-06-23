# Agent Orchestration System

This project demonstrates an agent orchestration system with two specialized agents and an orchestrator using LangGraph. The system mimics the Agent-to-Agent (A2A) protocol by having the orchestrator read agent information from a configuration file.

## Project Structure

```
.
├── agents/
│   ├── research_agent.py            # Research specialist agent
│   └── content_writing_agent.py     # Content writing specialist agent
├── config/
│   └── agent_registry.txt           # Agent registry file (A2A protocol simulation)
├── orchestrator/
│   └── orchestrator_agent.py        # Orchestrator agent using LangGraph
├── main.py                          # Main script to run all components
└── requirements.txt                 # Python dependencies
```

## How It Works

1. The system consists of three main components:
   - **Research Agent**: Specializes in information retrieval, fact checking, and literature review
   - **Content Writing Agent**: Specializes in content creation, editing, and summarization
   - **Orchestrator Agent**: Routes user queries to the appropriate specialized agent

2. The A2A protocol is simulated through the `agent_registry.txt` file, which contains:
   - Agent names and descriptions
   - Endpoints for each agent
   - Capabilities of each agent

3. When a user submits a query:
   - The orchestrator analyzes the query
   - It determines which specialized agent is best suited to handle the request
   - It forwards the query to the selected agent
   - The agent processes the query and returns a response
   - The orchestrator returns the final response to the user

## Architecture Diagram

```mermaid
graph TD
    subgraph User Interaction
        A[User] -->|1. Sends Query| B(main.py FastAPI);
    end

    subgraph Orchestration Layer
        B -->|2. Invokes| C{orchestrator_agent.py};
        C -->|3. Routes Query| D[Node: route_query];
        D -->|4. Reads Agent Info| E[config/agent_registry.txt];
        D -->|5. Selects Agent via AI| F(Gemini Model);
        C -->|6. Processes Query| G[Node: process_with_agent];
    end

    subgraph Agent Layer
        G -->|7a. HTTP POST Request| H(agents/research_agent.py);
        G -->|7b. HTTP POST Request| I(agents/content_writing_agent.py);
    end

    subgraph Data & AI Layer
        H -->|8a. Reads Data| J[data/*.csv];
        I -->|8b. Generates Content via AI| K(Gemini Model);
    end

    subgraph Response Flow
        H -->|9a. Returns Result| G;
        I -->|9b. Returns Result| G;
        G -->|10. Formats Response| C;
        C -->|11. Sends Final Response| B;
        B -->|12. Returns to User| A;
    end

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#bbf,stroke:#333,stroke-width:2px
    style C fill:#ccf,stroke:#333,stroke-width:2px
    style H fill:#fb9,stroke:#333,stroke-width:2px
    style I fill:#9fb,stroke:#333,stroke-width:2px
```

## Setup and Running

1. Ensure you have Python 3.8+ installed
2. Install the dependencies: `pip install -r requirements.txt`
3. Set your Gemini API key: `export GEMINI_API_KEY=your-gemini-key` (or use `set` on Windows)
4. Run the system: `python main.py`

This will start:
- Research Agent on port 8001
- Content Writing Agent on port 8002
- Orchestrator API on port 8000

## API Usage

Send a query to the orchestrator:

```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query":"Can you research the latest developments in renewable energy and write a summary?"}'
```

## LangGraph Implementation

The orchestrator uses LangGraph to create a workflow that:
1. Routes the query to the appropriate agent
2. Processes the query with the selected agent
3. Returns the response to the user

The workflow is defined as a state graph with decision nodes that determine the flow based on the query content.

## Extending the System

To add more specialized agents:
1. Create a new agent server implementation
2. Add the agent details to the `agent_registry.txt` file
3. Update the orchestrator logic if needed to handle the new agent capabilities
