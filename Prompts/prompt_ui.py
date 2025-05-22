from langchain_community.llms import HuggingFaceHub
from dotenv import load_dotenv
import streamlit as st
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline



load_dotenv()

model_path = "../tinyllama"
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
# model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, low_cpu_mem_usage=False)

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
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)



st.header(" Assistant")

user_input = st.text_area("Enter your  question here:")



if st.button("Submit"):
    prompt = f"Question: {user_input.strip()}\nAnswer:"
    result = generator(prompt, max_new_tokens=50, do_sample=False)
    answer = result[0]['generated_text'].replace(prompt, '').strip()
    st.write(answer)

