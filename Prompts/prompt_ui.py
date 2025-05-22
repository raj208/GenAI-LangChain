from langchain_community.llms import HuggingFaceHub
from dotenv import load_dotenv
import streamlit as st
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline



load_dotenv()

model_path = "../tinyllama"
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,
    # temperature=0,
    do_sample=True
    # eos_token_id=tokenizer.eos_token_id
)

llm = HuggingFacePipeline(pipeline=pipe)



st.header("Research Assistant")

user_input = st.text_area("Enter your research question here:")



if st.button("Submit"):
    result = llm.invoke(user_input)
    st.write(result)
