from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-0.6B",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

st.header("Research Assistant")

user_input = st.text_input("Enter your question here:")

if st.button('summarize'):
    result = model.invoke(user_input)
    st.write(result.content)
