from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-235B-A22B",
    task="text-generation",
        pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=10
    )
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What do you mean by sopohomore ?")

print(result.content)