from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-0.6B",
    task="text-generation",
    max_new_tokens=32,  
    timeout=300
)

model = ChatHuggingFace(llm=llm)

try:
    result = model.invoke("What is the capital of India")
    if result is None or result.content is None:
        print("Error: No response from Hugging Face endpoint. Check your API key and model availability.")
    else:
        print(result.content)
except Exception as e:
    print("Error:", e)
