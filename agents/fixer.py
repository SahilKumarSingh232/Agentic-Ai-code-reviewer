
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from rag.retriever import KnowledgeRetriever


class FixerAgent:

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
                You are an expert Python code fixing engineer.

                Your responsibility is to fix issues found
                during code review.

                You have access to a knowledge base containing
                Python programming guidelines and best practices.

                Use the retrieved knowledge to guide your
                code improvements.

                Instructions:

                1. Fix logical errors.
                2. Fix runtime errors.
                3. Improve variable and function names.
                4. Follow Python PEP 8 conventions.
                5. Improve readability.
                6. Handle important edge cases.
                7. Preserve the original functionality.
                8. Add documentation where useful.

                IMPORTANT:

                - Analyze the original code carefully.
                - Use retrieved knowledge only when relevant.
                - Fix the issues provided in the review.
                - Do not introduce unnecessary modifications.
                - Do not invent missing requirements.
                - Preserve the intended behavior of the program.
                - Return the complete corrected Python code.
                - Do not include explanations outside the code.
                - Return only Python code inside a code block.
                """
            ),
            (
                "human",
                """
                Retrieved Python Knowledge:

                {knowledge}

                --------------------------------

                Original Python Code:

                {code}

                --------------------------------

                Issues Found During Review:

                {issues}

                --------------------------------

                Generate the complete corrected Python code
                using the relevant retrieved knowledge.
                """
            )
        ])

        # Create LangChain LCEL pipeline
        self.chain = self.prompt | self.llm

    def fix(self, code, issues):

        # Retrieve relevant Python knowledge from ChromaDB
        retrieved_results = self.retriever.retrieve(
            query=code + "\n" + issues,
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

        # Send code, issues, and retrieved knowledge to the LLM
        response = self.chain.invoke({
            "code": code,
            "issues": issues,
            "knowledge": knowledge
        })

        return response.content


# Test the Fixer Agent
if __name__ == "__main__":

    agent = FixerAgent()

    sample_code = """
def CALCULATE(x,y):
  a=x+y
  b=a*2
  return b

print(CALCULATE(10,20))
"""

    sample_issues = """
    1. Function name should use snake_case.
    2. Variable names should be descriptive.
    3. Formatting should follow PEP 8.
    4. Add a useful docstring.
    """

    print("\n===== FIXING CODE WITH RAG =====")

    result = agent.fix(sample_code, sample_issues)

    print("\n===== FIXED CODE =====\n")
    print(result)
