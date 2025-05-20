from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline

# Load from your local path
model_path = "./tinyllama"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Create text-generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,
    # temperature=0,
    do_sample=True
    # eos_token_id=tokenizer.eos_token_id
)



# Wrap in LangChain pipeline wrapper
llm = HuggingFacePipeline(pipeline=pipe)

# Directly invoke the LLM (you don't need ChatHuggingFace)
prompt = "[INST] What is the capital of India? [/INST]"
result = llm.invoke(prompt)
print(result)

