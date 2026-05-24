# check langchain version if not uncomment below pip command to install the correct version
import langchain, langchain_community
import os # For setting environment variables in utils.py
import warnings # To ignore any warnings during execution, you can use the warnings library to filter them out. This is optional but can help keep the output clean.
from dotenv import load_dotenv 

# Importing necessary libraries from LangChain for creating agents and tools
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.agents import AgentExecutor, create_react_agent, create_tool_calling_agent
from langchain.tools import tool


# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


warnings.filterwarnings('ignore')


# The LLM (Agent Brain)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Define Tool Using @tool Decorator
@tool
def multiply_numbers(query: str) -> str:
    """
    Multiply two numbers found in a text input.
    Input: string containing two numeric values.
    Output: numeric multiplication result.
    """
    try:
        nums = [float(x) for x in query.split() if x.replace('.', '', 1).isdigit()]
        return str(nums[0] * nums[1])
    except:
        return "Couldn't extract numeric values."

@tool
def add_numbers(query: str) -> str:
    """
    Add two numbers found in the text input.
    Input: string containing two numeric values.
    Output: numeric addition result.
    """
    try:
        nums = [float(x) for x in query.split() if x.replace('.', '', 1).isdigit()]
        return str(nums[0] + nums[1])
    except:
        return "Couldn't extract numeric values."
    
@tool
def subtract_numbers(query: str) -> str:
    """
    Subtract two numbers found in the text input.
    Input: string containing two numeric values.
    Output: numeric addition result.
    """
    try:
        nums = [float(x) for x in query.split() if x.replace('.', '', 1).isdigit()]
        return str(nums[0] - nums[1])
    except:
        return "Couldn't extract numeric values."



# Tools list
tools = [multiply_numbers,add_numbers,subtract_numbers]

# Tool-Calling Prompt 
prompt = ChatPromptTemplate.from_template("""
You are a helpful reasoning AI agent.

Follow this decision process:
1. Understand the user request.
2. Decide whether a tool is needed.
3. If yes → call the correct tool.
4. If no → answer directly.
5. Reflect before final output.


User Query: {input},
("placeholder", "{agent_scratchpad}")
""")

# Create the Agent with Tools
agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# Agent Runtime using AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=1 )

# Test agent
result = agent_executor.invoke({"input": "Subtract 15 and 4"})
print("\nFINAL OUTPUT:", result["output"])