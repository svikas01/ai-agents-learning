'''
Simple program to demonstrate how to make a basic call 
to a Large Language Model (LLM) using the LangChain library and the Groq API. 
This example will read the API key from a .env file, initialize the LLM, 
and make a test call to generate a response based on a given prompt.
'''

from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# 1. Load .env file (must be in root)
load_dotenv()

# 2. Read the key (debug line)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print("GROQ_API_KEY =", GROQ_API_KEY)  # Check if it is read correctly

# 3. Check if the key is present
if not GROQ_API_KEY:
    raise EnvironmentError("GROQ_API_KEY is not set in .env file.")

# 3. Initialize the LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY,
    temperature=0.7
)

# 4. Make a test call
#prompt = "Say 'Hello Vikas' and tell me a short fact about Python."
prompt = "Say 'Hello Travel Enthusiast' and tell a interesting fact about Bali, Indonesia."

response = llm.invoke(prompt)
print("LLM Response:")
print("----------------")
print(response.content)