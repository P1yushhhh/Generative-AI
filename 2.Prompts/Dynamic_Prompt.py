from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-0.6B",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

st.header("Research Assistant")

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

# template = load_prompt('template.json')
template = load_prompt(r'Prompts\template.json')
# fill the placeholders in the template with user inputs
prompt = template.invoke({
    "paper_input": paper_input,
    "style_input": style_input,
    "length_input": length_input
})

if st.button("Generate Summary"):
    with st.spinner("Generating summary..."):
        response = model.invoke(prompt)
        st.write(response.content)
        