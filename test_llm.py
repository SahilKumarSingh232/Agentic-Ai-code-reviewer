
from langchain_ollama import ChatOllama

# Connect to Qwen2.5-Coder 7B
llm = ChatOllama(
    model="qwen2.5-coder:7b",
    temperature=0
)

# Test the model
response = llm.invoke(
    "Find the bug in this Python code: print(10 / 0)"
)

print("\nAI RESPONSE:")
print(response.content)
