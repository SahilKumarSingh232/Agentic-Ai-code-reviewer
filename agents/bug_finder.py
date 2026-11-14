
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from rag.retriever import KnowledgeRetriever


class BugFinderAgent:

    def __init__(self):

        # Initialize the local LLM
        self.llm = ChatOllama(
            model="qwen2.5-coder:7b",
            temperature=0
        )

        # Initialize RAG retriever
        self.retriever = KnowledgeRetriever()

        # Define the prompt
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                You are an expert Python debugging engineer.

                Your responsibility is to identify bugs in Python code.

                You have access to a knowledge base containing
                Python programming guidelines and debugging knowledge.

                Use the retrieved knowledge as supporting reference
                material while analyzing the submitted code.

                Check for:
                1. Logical errors
                2. Runtime errors
                3. Edge cases
                4. Incorrect conditions
                5. Potential security risks

                For every issue:
                - Mention the line or code section.
                - Explain the problem.
                - Explain why it is a problem.
                - Suggest how to fix it.

                Important instructions:

                - Analyze the actual submitted code.
                - Use retrieved knowledge only when relevant.
                - Do not invent bugs just to match the knowledge base.
                - Do not assume an error exists without evidence.
                - Distinguish actual bugs from potential risks.
                - If no bugs are found, clearly mention that.
                - Do not rewrite the entire code.
                - Focus only on identifying and explaining bugs.
                """
            ),
            (
                "human",
                """
                Retrieved Python Knowledge:

                {knowledge}

                --------------------------------

                Submitted Python Code:

                {code}

                --------------------------------

                Analyze the submitted code using the relevant
                retrieved knowledge.
                """
            )
        ])

        # Create LangChain LCEL pipeline
        self.chain = self.prompt | self.llm

    def analyze(self, code):

        # Retrieve relevant knowledge from ChromaDB
        retrieved_results = self.retriever.retrieve(
            query=code,
            top_k=3
        )

        # Prepare retrieved knowledge
        if retrieved_results:

            knowledge = "\n\n".join(
                [
                    f"Category: {result['category']}\n"
                    f"Content: {result['content']}"
                    for result in retrieved_results
                ]
            )

        else:

            knowledge = "No relevant knowledge was retrieved."

        # Send code and retrieved knowledge to the LLM
        response = self.chain.invoke({
            "code": code,
            "knowledge": knowledge
        })

        return response.content


# Test the Bug-Finder Agent
if __name__ == "__main__":

    agent = BugFinderAgent()

    sample_code = """
def divide(a, b):
    return a / b

print(divide(10, 0))
"""

    print("\n===== ANALYZING CODE WITH RAG =====")

    result = agent.analyze(sample_code)

    print("\n===== BUG FINDER REPORT =====\n")
    print(result)
