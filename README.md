# Intelligent Text-to-SQL Agent with Deep Schema Awareness

This project is a sophisticated AI agent designed to translate natural language questions into accurate, executable SQL queries. It is particularly powerful when working with complex databases that have intricate relationships, metadata, and business-specific logic.

## Vision

The goal of this project is to create an intelligent and autonomous "database architect" agent. Instead of simply translating words to SQL, this agent first learns the structure and relationships within a database. By building its own internal "knowledge map," it can reason about how to best answer a user's question, resulting in more accurate and efficient queries, especially for complex schemas.

## Key Features

- **Intelligent Agentic Workflow:** Utilizes a multi-agent system (powered by LangGraph) where specialized agents collaborate on planning, SQL generation, and code review.
- **Deep Schema Awareness:** Before generating a query, the agent first runs a `DatabaseDiscoveryService` to analyze the database's `information_schema`. It learns the tables, columns, data types, and—most importantly—the foreign key relationships.
- **Performance Caching:** The discovered database schema is cached locally in a JSON file, ensuring fast startup times and minimizing redundant database introspection.
- **Context-Aware Prompts:** The complete database schema map is dynamically injected into the AI's prompt, giving it the full structural context needed to write accurate `JOIN`s and complex queries.
- **Multi-LLM Support:** Easily configurable to use various Large Language Models, including those from OpenAI, Groq, and local models via Ollama.
- **Interactive UI:** A simple and intuitive chat interface built with Streamlit allows for easy interaction with the agent.

## How It Works

The agent's intelligence comes from a structured, multi-step process that transforms a user's question into a correct SQL query.

1.  **Schema Discovery:** On startup, the `PlannerAgent` initializes a `DatabaseDiscoveryService`. This service connects to the PostgreSQL database and builds a detailed "schema map" of all tables, columns, and their relationships. This map is then cached to `database/schema_cache.json`.
2.  **Contextual Planning:** When a user asks a question (e.g., "Which interfaces are down?"), the `PlannerAgent` loads the schema map from the cache and injects it into its prompt to the LLM. With this full context, it creates a high-level plan for how to answer the question.
3.  **SQL Generation:** The plan is passed to the `SQLGenerator` agent, which uses the plan and its knowledge of the schema to write the actual SQL query.
4.  **Review and Refinement:** The generated SQL is reviewed by the `ReviewerAgent` to check for correctness, security, and performance.
5.  **Execution & Reporting:** The final, approved query is executed against the database, and the results are presented to the user in a clear, summarized report.

## Getting Started

### Prerequisites

- Anaconda (or another Python environment manager)
- Python 3.11+

### 1. Environment Setup

1.  **Create and Activate a Virtual Environment:**
    ```bash
    conda create -n agent_env python=3.11 pip
    conda activate agent_env
    ```

2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/john-adeojo/graph_websearch_agent.git
    cd graph_websearch_agent
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### 2. Configure API Keys

The application requires API keys for the desired Large Language Model. These are configured in `config.yaml`.

1.  **Open the `config.yaml` file:**
    ```bash
    nano config.yaml
    ```

2.  **Enter your API keys:**
    - **OpenAI API Key:** Get from [https://openai.com/](https://openai.com/)
    - **Groq API Key:** Get from [https://console.groq.com/keys](https://console.groq.com/keys)
    - *(Add other keys as needed, e.g., for Gemini, Claude, etc.)*

### 3. Running the Application

Once configured, you can run the Streamlit-based user interface.

-   **For Windows:**
    ```powershell
    run_windows.ps1
    ```
-   **For Linux/macOS:**
    ```bash
    chmod +x run_linux.sh
    ./run_linux.sh
    ```

### (Optional) Using a Local LLM with Ollama

For development and offline use, you can run the agent with a local model via Ollama.

1.  **Download and Install Ollama:**
    [https://ollama.com/download](https://ollama.com/download)

2.  **Pull a Model:**
    Open a new terminal and pull your desired model (e.g., `llama3`).
    ```bash
    curl http://localhost:11432/api/pull -d "{\"name\": \"llama3\"}"
    ```

3.  **Configure the App:**
    In the Streamlit UI, select "ollama" as the server and choose your local model from the dropdown.
