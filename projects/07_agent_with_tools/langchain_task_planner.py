from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import os # For setting environment variables in utils.py
from dotenv import load_dotenv 

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Create a prompt template for the agent
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert task planner. Break down complex tasks into manageable steps."),
    ("human", "{task_description}")
])

# set up the LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5
    )

# set up the chain
chain = prompt | llm

task_description = "Plan a 9 day trip to bali focusing on cultural experiences."
response = chain.invoke({"task_description": task_description})
print(response.content)