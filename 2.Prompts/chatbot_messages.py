from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-0.6B",
    task="text-generation",
    max_new_tokens=32,  
    timeout=300
)
model = ChatHuggingFace(llm=llm)

chat_history = [
    SystemMessage(content="You are a helpful assistant."),
    SystemMessage(content="Only answer the user's question directly. Do not show your reasoning or use <think> tags."),
]

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input == "exit":
        print("Exiting the chatbot. Goodbye!")
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("Chatbot:", result.content)

print("Chat history:", chat_history)