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

chat_history = []



while True:
    user_input = input("You: ")
    chat_history.append(user_input)
    if user_input == "exit":
        print("Exiting the chatbot. Goodbye!")
        break
    result = model.invoke(chat_history)
    chat_history.append(result.content)
    print("Chatbot:", result.content)

print("Chat history:", chat_history)
