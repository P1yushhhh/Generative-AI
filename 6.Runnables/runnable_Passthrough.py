from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableParallel, RunnableSequence, RunnablePassthrough
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class PoemOutput(BaseModel):
    poem: str = Field(description="The generated poem")
    explanation: str = Field(description="Explanation of the poem")

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "meta-llama/Llama-3.1-8B-Instruct",
    task = "text-generation"
)
model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Generate a short poem about {topic}.",
    input_variables=["topic"]
)

parser2 = PydanticOutputParser(pydantic_object=PoemOutput)

prompt2 = PromptTemplate(
    template="Generate a 5 line explanation about {text}.\n{format_instructions}",
    input_variables=["text"],
    partial_variables={"format_instructions": parser2.get_format_instructions()}
)


chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    'Poem': RunnablePassthrough(),
    'Explanation': RunnableSequence(prompt2, model, parser2)
})

final_chain = RunnableSequence(chain, parallel_chain)

print(final_chain.invoke({'topic' : 'monsoon'}))