# 🧮 AI-Powered Math Agent

A conversational AI agent that solves arithmetic problems expressed in plain English. Built with **LangChain** and **OpenAI's GPT-4o-mini**, this agent interprets your math question, selects the appropriate calculator tools, executes the operations step by step, and returns the final answer.

---

## 📌 Table of Contents

- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Code Walkthrough](#code-walkthrough)
  - [1. Importing Libraries](#1-importing-libraries)
  - [2. Loading the API Key](#2-loading-the-api-key)
  - [3. Creating the Prompt Template](#3-creating-the-prompt-template)
  - [4. Defining the Math Tools](#4-defining-the-math-tools)
  - [5. Creating the AI Model](#5-creating-the-ai-model)
  - [6. Building the Agent](#6-building-the-agent)
  - [7. Creating the Executor](#7-creating-the-executor)
  - [8. Running the Agent](#8-running-the-agent)
- [Example Output](#example-output)
- [Things to Watch Out For](#things-to-watch-out-for)


---

## How It Works

```
User asks a math question in plain English
        ↓
   AI Agent (GPT-4o-mini) reads the question
        ↓
   Agent decides WHICH math tool(s) to use
        ↓
   Tools execute the actual math calculations
        ↓
   Agent compiles the results and responds
```

The agent follows a **think → act → observe** loop. It reads the user's question, decides which tool to call, reads the tool's output, and then decides whether to call another tool or return a final answer.

---

## Tech Stack

| Technology | Purpose |
|---|---|
| [Python 3.x](https://www.python.org/) | Programming language |
| [LangChain](https://www.langchain.com/) | Framework for building AI agents |
| [OpenAI GPT-4o-mini](https://platform.openai.com/) | Large Language Model (the agent's "brain") |
| [python-dotenv](https://pypi.org/project/python-dotenv/) | Loads environment variables from `.env` file |

---

## Project Structure

```
ai-math-agent/
├── .env                # Stores your OpenAI API key (DO NOT commit this)
├── .gitignore          # Ensures .env and other sensitive files are not pushed
├── main.py             # Main application code
├── requirements.txt    # Python dependencies
└── README.md           # This documentation file
```

---

## Setup & Installation

### Prerequisites

- Python 3.9 or higher
- An OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### Steps

**1. Clone the repository**

```bash
git clone https://github.com/your-username/ai-math-agent.git
cd ai-math-agent
```

**2. Create a virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Create the `.env` file**

Create a file named `.env` in the project root and add your OpenAI API key:

```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
```

> ⚠️ **Never commit your `.env` file to GitHub.** Make sure `.env` is listed in your `.gitignore`.

**5. Run the application**

```bash
python main.py
```

---

## Code Walkthrough

### 1. Importing Libraries

```python
import os
from dotenv import load_dotenv
```

- `os` — A built-in Python library used to read environment variables from the operating system.
- `dotenv` / `load_dotenv` — Reads key-value pairs from a `.env` file and loads them into the environment so the code can access secrets (like API keys) without hardcoding them.

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
```

- `ChatPromptTemplate` — A class that lets you define a structured conversation template (system instructions, user input, agent working memory) that the AI model follows every time it runs.

```python
from langchain_openai import ChatOpenAI
```

- `ChatOpenAI` — LangChain's wrapper around OpenAI's chat models. It handles all communication with OpenAI's API servers behind the scenes.

```python
from langchain.tools import tool
```

- `@tool` — A Python **decorator**. When you place `@tool` above a function, it registers that function as a tool the AI agent is allowed to call. The agent reads the function's name, parameters, and docstring to understand when and how to use it.

```python
from langchain_classic.agents import AgentExecutor, create_react_agent, create_tool_calling_agent
```

- `create_tool_calling_agent` — A factory function that builds an AI agent capable of calling tools using OpenAI's native tool-calling feature.
- `AgentExecutor` — The runtime engine that manages the agent's think → act → observe loop.
- `create_react_agent` — Imported but **not used** in this code. Can be removed.

---

### 2. Loading the API Key

```python
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
```

This reads the `.env` file and extracts the `OPENAI_API_KEY` value. This key authenticates your application with OpenAI's servers. Without a valid key, all API calls will fail.

---

### 3. Creating the Prompt Template

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert mathematician. Use the tools provided to solve arithmetic problems accurately."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])
```

This defines the conversation structure the agent uses on every run. It has three parts:

| Part | Role | Description |
|---|---|---|
| `system` | AI's persona & rules | Tells the AI it is a math expert and must use the provided tools. This instruction persists for the entire conversation. |
| `human` / `{input}` | User's question | A placeholder that gets replaced at runtime with the actual math question the user types. |
| `placeholder` / `{agent_scratchpad}` | Agent's working memory | Stores intermediate tool calls and results as the agent works through multi-step problems. Think of it as the agent's scratch paper. |

---

### 4. Defining the Math Tools

Each tool is a standard Python function decorated with `@tool`. The agent reads three things from each tool to decide when and how to use it:

- **Function name** — identifies the tool (e.g., `add_number`).
- **Parameter types** — tells the agent what inputs are needed (e.g., `x: float, y: float`).
- **Docstring** — the description the agent reads to understand the tool's purpose. **This is critical** — a vague or missing docstring can cause the agent to pick the wrong tool.

```python
@tool
def add_number(x: float, y: float) -> float:
    """Add two numbers."""
    return x + y

@tool
def substract_number(x: float, y: float) -> float:
    """Substract two numbers."""
    return x - y

@tool
def multiply_number(x: float, y: float) -> float:
    """Multiply two numbers."""
    return x * y

@tool
def divide_number(x: float, y: float) -> float:
    """Divide two numbers."""
    if y == 0:
        return "Error: Division by zero is undefined."
    return x / y

@tool
def square_number(x: float) -> float:
    """Square a number."""
    return x * x
```

#### Tool Reference

| Tool | What It Does | Inputs | Example |
|---|---|---|---|
| `add_number` | Adds two numbers | `x`, `y` | `add_number(5, 3)` → `8` |
| `substract_number` | Subtracts `y` from `x` | `x`, `y` | `substract_number(10, 4)` → `6` |
| `multiply_number` | Multiplies two numbers | `x`, `y` | `multiply_number(3, 7)` → `21` |
| `divide_number` | Divides `x` by `y` (with zero-check) | `x`, `y` | `divide_number(10, 2)` → `5.0` |
| `square_number` | Squares a single number | `x` | `square_number(4)` → `16` |

> **Note:** `divide_number` includes a safety check — if the second number (`y`) is zero, it returns an error message instead of crashing the program.

All tools are then collected into a list:

```python
tools = [add_number, substract_number, divide_number, multiply_number, square_number]
```

---

### 5. Creating the AI Model

```python
model = ChatOpenAI(model="gpt-4o-mini")
```

This creates an instance of OpenAI's **GPT-4o-mini** model. It is a smaller, faster, and more cost-effective version of GPT-4o. This model acts as the "brain" that reads the user's question and decides which tools to call. It automatically picks up the API key from the environment variables.

---

### 6. Building the Agent

```python
agent = create_tool_calling_agent(
    llm=model,
    tools=tools,
    prompt=prompt
)
```

This assembles the agent by combining three ingredients:

- `llm` — the AI model (GPT-4o-mini)
- `tools` — the list of math functions the agent can call
- `prompt` — the conversation template

The `create_tool_calling_agent` function uses OpenAI's **native tool-calling** capability, meaning the model outputs structured requests like _"call `add_number` with `x=5, y=3`"_ rather than trying to compute the answer itself.

---

### 7. Creating the Executor

```python
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)
```

The `AgentExecutor` is the runtime engine that runs the agent's reasoning loop:

```
Think → Pick a tool → Call the tool → Read the result → Think again → Repeat until done
```

The `verbose=True` flag makes the executor print every intermediate step to the console. This is very helpful for debugging — you can see exactly which tools the agent chose, what inputs it provided, and what results came back. Set this to `False` in production to suppress the logs.

---

### 8. Running the Agent

```python
new_output = executor.invoke({
    "input": "What is the result of adding 5 and 3, then multiplying by 2, and finally raising to the power of 2?"
})
print("New Agent Output:", new_output)
```

The `invoke()` method sends the user's question to the agent. The `"input"` key in the dictionary corresponds to the `{input}` placeholder defined in the prompt template.

For this particular question, the agent internally performs these steps:

1. Calls `add_number(5, 3)` → gets `8`
2. Calls `multiply_number(8, 2)` → gets `16`
3. Calls `square_number(16)` → gets `256`
4. Returns the final answer: **256**

---

## Example Output

When you run the script with `verbose=True`, you will see output similar to this:

```
> Entering new AgentExecutor chain...

Invoking: `add_number` with `{'x': 5.0, 'y': 3.0}`
8.0

Invoking: `multiply_number` with `{'x': 8.0, 'y': 2.0}`
16.0

Invoking: `square_number` with `{'x': 16.0}`
256.0

The result of adding 5 and 3 (which gives 8), then multiplying by 2
(which gives 16), and finally squaring (which gives 256) is **256**.

> Finished chain.

New Agent Output: {'input': '...', 'output': 'The result is **256**.'}
```

---

## Things to Watch Out For

### 🔑 API Key Security
Never hardcode your API key in the source code. Always use a `.env` file and ensure it is listed in `.gitignore`.

### 💰 API Costs
Every call to `executor.invoke()` makes one or more API requests to OpenAI. Multi-step problems make multiple calls. Monitor your usage on the [OpenAI dashboard](https://platform.openai.com/usage).

### 📦 LangChain Versioning
This code imports from `langchain_classic`, which indicates a specific or pinned version of LangChain. LangChain updates frequently and often introduces breaking changes. Always match the exact package versions from `requirements.txt`.

### ⚠️ No Error Handling
The current code does not handle exceptions. If the API key is missing, the network is down, or OpenAI's servers are unavailable, the script will crash. For production use, wrap the `invoke()` call in a `try/except` block.

### 🧰 Limited Tool Set
The agent only has basic arithmetic tools (add, subtract, multiply, divide, square). If a user asks for operations like square root, exponentiation to an arbitrary power, or trigonometry, the agent has no tool for it and may produce incorrect results or decline to answer. Extend the `tools` list to support more operations.

---
