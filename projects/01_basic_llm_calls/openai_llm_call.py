'''
Simple program to demonstrate how to make a basic call 
to a Large Language Model (LLM) using the LangChain library and the OpenAI API. 
This example will read the API key from a .env file, initialize the LLM, 
and make a test call to generate a response based on a given prompt.
'''

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# 1. Load .env file (must be in root)
load_dotenv()

# 2. Read the key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 3. Check if the key is present
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY is not set in .env file.")

# 4. Initialize the LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY,
    temperature=0.7
)

# 5. Make a test call
prompt = "Say 'Hello Travel Enthusiast' and tell an interesting fact about Bali, Indonesia."

response = llm.invoke(prompt)
print("LLM Response:")
print("----------------")
print(response.content)