from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = 'meta-llama/Meta-Llama-3-8B-Instruct',
    task = 'text-generation'
)

model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()

loader = TextLoader('7.DocumentLoaders\Cricket.txt', encoding = 'utf-8')

prompt = PromptTemplate(
    template = 'Write a summary for the following poem {text}',
    input_variables = ['text']
)

docs = loader.load()

print(docs[0].page_content)
print(docs[0].metadata)

print(type(docs))

print(len(docs))

print(type(docs[0]))

chain = prompt | model | parser

result = chain.invoke({'text': docs[0].page_content})

print(result)