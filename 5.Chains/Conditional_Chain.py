from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
)
model1 = ChatHuggingFace(llm=llm)

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description="The sentiment of the text")

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template = "Classify the sentiment of the following text as positive or negative \n {text} \n {format_instructions}",
    input_variables = ["text"],
    partial_variables = {'format_instructions' : parser2.get_format_instructions() } 
)
parser = StrOutputParser()

classifier_chain = prompt1 | model1 | parser2

result = classifier_chain.invoke({"text": " This is a wonderfull smartphone!"})

print(result)

prompt2 = PromptTemplate(
    template = "Write an appropriate response to the following positive feedback \n {text}",
    input_variables = ["text"]
)

prompt3 = PromptTemplate(
    template = "Write an appropriate response to the following negative feedback \n {text}",
    input_variables = ["text"]
)


branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | model1 | parser),
    (lambda x:x.sentiment == 'negative', prompt3 | model1 | parser),
    RunnableLambda(lambda x: "could not classify the sentiment of the text")
)

chain = classifier_chain | branch_chain

result = chain.invoke({"text": " This is a wonderfull smartphone!"})
print(result)

chain.get_graph().print_ascii()
