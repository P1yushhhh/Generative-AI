from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
)
model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template = "Explain the concept of {concept} in 1-2 paragraph",
    input_variables = ["concept"]
)

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template = "List 5 key points about {text}",
    input_variables = ["text"]
)

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({"concept": "Agentic AI"})

print(result)

chain.get_graph().print_ascii()