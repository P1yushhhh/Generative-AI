from langchain_core.prompts import PromptTemplate, load_prompt
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-0.6B",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

template = PromptTemplate(
    template = " Greet this person in 5 languages when the name of the person is {name}",
    input_variables=["name"]
)

prompt = template.invoke({ 'name' : 'Piyush'})

result = model.invoke(prompt)

print(result.content)