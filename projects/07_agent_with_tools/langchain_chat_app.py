from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import os # For setting environment variables in utils.py
from dotenv import load_dotenv 

# Create a prompt template for the agent
prompt = ChatPromptTemplate.from_messages([
    ("system", "you are a helpful AI tutor. Respond to student questions clearly and shortest way possible"),
    ("human", "{input}")
])

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# set up the LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5
    )

# set up the chain
chain = prompt | llm

# Interactive chat
print("AI Tutor: Hello! Ask me anything. Type 'quit' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        print("AI Tutor: Goodbye!")
        break
    response = chain.invoke({"input": user_input})
    print(f"AI Tutor: {response.content}")