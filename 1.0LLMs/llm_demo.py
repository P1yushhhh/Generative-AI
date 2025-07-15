from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv() # load key from the .env file 

llm = OpenAI(model = '') #specify the model you want to use, e.g., "gpt-3.5-turbo"

result = llm.invoke("Who is the president of USA") #ask the question you want the llm to answer

print(result)