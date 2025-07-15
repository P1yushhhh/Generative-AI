from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

#chat template
chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{queury}'),
])

chat_history = []
#laod chat history
with open('Prompts/Chathistory.txt', 'r') as file:
    chat_history.extend(file.readlines())

print(chat_history)

#create prompt
prompt = chat_template.invoke({
    'chat_history': chat_history,
    'queury': 'What is the status of my order?'
    })
