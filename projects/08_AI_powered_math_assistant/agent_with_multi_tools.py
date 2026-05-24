## IMPORTING LIBRARIES
# For setting environment variables in utils.py
import os 
from dotenv import load_dotenv 

# Library to create Chat Prompt
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

# Library to create agent using Open AI
from langchain_openai import ChatOpenAI

# Library to create tools for the agent, provide @tool decorator
from langchain.tools import tool


from langchain_classic.agents import AgentExecutor, create_tool_calling_agent


# Loading environment variables to get the OpenAI API key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Creating a prompt template for the agent
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert mathematician. Use the tools provided to solve arithmetic problems accurately."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])


# Defining the tools that the agent can use
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


tools = [add_number, substract_number, divide_number, multiply_number, square_number]

# Create model
model = ChatOpenAI(model="gpt-4o-mini")

agent = create_tool_calling_agent(
    llm=model, 
    tools=tools, 
    prompt=prompt)

executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)


new_output = executor.invoke({"input": "What is the result of adding 5 and 3, then multiplying by 2, and finally raising to the power of 2?"})# --- IGNORE ---
print("New Agent Output:", new_output)
