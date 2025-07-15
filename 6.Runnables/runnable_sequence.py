from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence

load_dotenv()

prompt = PromptTemplate(
    template = "Translate the following English text to french {input}",
    input_variables = ["input"]
)

llm = HuggingFaceEndpoint(
    repo_id = "meta-llama/Llama-3.1-8B-Instruct",
    task = "text-generation"
)

prompt2 = PromptTemplate(
    template = "Now Translate the following French text to English {input}",
    input_variables = ["input"]
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

chain = RunnableSequence(prompt, model, parser)
chain2 = RunnableSequence(prompt2, model, parser)

# First translation: English to French
french_output = chain.invoke({"input": "I love programming in Python"})
print("English to French:", french_output)

# Second translation: French back to English
english_output = chain2.invoke({"input": french_output})
print("French to English:", english_output)