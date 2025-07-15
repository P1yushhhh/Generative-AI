from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model = "gpt-4.1-nano-2025-04-14", temperature = 0, max_completion_tokens = 20)

result = model.invoke("suggest me a name for a sotrybook character")
print(result.content)