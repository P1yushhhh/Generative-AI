from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableParallel

load_dotenv()

llm1 = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
)
model1 = ChatHuggingFace(llm=llm1)

llm2 = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
)   
model2 = ChatHuggingFace(llm=llm2)

prompt1 = PromptTemplate(
    template="Explain the concept of {concept} in 1-2 paragraph",
    input_variables=["concept"]
)

prompt2 = PromptTemplate(
    template="Generate a quiz with 5 questions about {concept} and provide the answers.",
    input_variables=["concept"]
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz in a single document \n notes -> {notes} \n quiz -> {quiz}.",
    input_variables=["notes", "quiz"]
)

parser = StrOutputParser()  

parallel_chain = RunnableParallel(
    {
        'notes' : prompt1 | model1 | parser,
        'quiz' : prompt2 | model2 | parser
    }
)
merged_chain = prompt3 | model1 | parser

chain = parallel_chain | merged_chain

text = """
Agentic AI refers to artificial intelligence systems that possess a degree of autonomy and decision-making capabilities, allowing them to act independently in pursuit of specific goals. These systems can analyze their environment, make choices based on learned experiences, and adapt their strategies to achieve desired outcomes. Agentic AI is often characterized by its ability to learn from interactions, optimize performance over time, and operate in dynamic and complex environments. Examples include autonomous robots, intelligent virtual assistants, and self-learning algorithms used in various applications such as robotics, finance, and healthcare. The development of agentic AI raises important ethical and safety considerations, as these systems can make decisions that impact human lives and society at large. It is crucial to ensure that such AI systems are designed with transparency, accountability, and alignment with human values to mitigate potential risks and ensure beneficial outcomes.
With the rise of agentic AI, there is a growing need for frameworks and guidelines to govern their behavior, ensuring they operate within ethical boundaries and do not cause harm. This includes establishing protocols for decision-making, risk assessment, and human oversight. As agentic AI continues to evolve, it holds the potential to revolutionize industries by automating complex tasks, enhancing productivity, and enabling new forms of human-machine collaboration. As these systems become more prevalent, ongoing research and dialogue will be essential to address the challenges and opportunities they present, fostering a future where agentic AI contributes positively to society while minimizing risks.
"""

result = chain.invoke({"concept": text})

print(result)

chain.get_graph().print_ascii()
