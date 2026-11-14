
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from rag.retriever import KnowledgeRetriever


class StyleLintAgent:

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
                You are an expert Python code quality reviewer.

                Your responsibility is to review coding style
                and readability.

                You have access to a knowledge base containing
                Python coding standards and best practices.

                Use the retrieved knowledge as supporting
                reference material during your review.

                Check for:

                1. Variable and function naming conventions.
                2. Code readability.
                3. Proper indentation and formatting.
                4. Unnecessary or duplicate code.
                5. PEP 8 coding conventions.
                6. Missing comments or documentation where useful.
                7. Code organization and maintainability.

                For every issue:
                - Identify the problematic code.
                - Explain what is wrong.
                - Suggest an improvement.

                Give a short overall code quality assessment.

                IMPORTANT:

                - Analyze the actual submitted code.
                - Use retrieved knowledge only when relevant.
                - Do not invent style issues.
                - Do not report the same issue repeatedly.
                - Do not focus on logical bugs or runtime errors.
                - Do not rewrite the entire code.
                - Focus only on coding style and quality.
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

                Review the coding style using the relevant
                retrieved knowledge.
                """
            )
        ])

        # Create LangChain LCEL pipeline
        self.chain = self.prompt | self.llm

    def review(self, code):

        # Retrieve relevant coding knowledge from ChromaDB
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

        # Send retrieved knowledge and code to the LLM
        response = self.chain.invoke({
            "code": code,
            "knowledge": knowledge
        })

        return response.content


# Test the Style/Lint Agent
if __name__ == "__main__":

    agent = StyleLintAgent()

    sample_code = """
def CALCULATE(x,y):
  a=x+y
  b=a*2
  return b

print(CALCULATE(10,20))
"""

    print("\n===== ANALYZING CODE WITH RAG =====")

    result = agent.review(sample_code)

    print("\n===== STYLE / LINT REPORT =====\n")
    print(result)
