from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load from local folder or model ID
model_path = "./tinyllama"  # or use model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Create a text generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Run inference
prompt = "What is the capital of India?"
output = generator(prompt, max_new_tokens=50, do_sample=True)

print(output[0]["generated_text"])
