
import json
import re

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from rag.retriever import KnowledgeRetriever


class TestWriterAgent:

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
                You are an expert Python software testing engineer.

                You have access to a knowledge base containing
                Python testing guidelines and best practices.

                Use the retrieved knowledge as supporting
                reference material when generating test scenarios.

                Your responsibility is to generate accurate test
                scenarios for the given Python source code.

                Follow these instructions carefully:

                1. Identify all actual functions in the source code.
                2. Generate normal test cases.
                3. Generate edge cases.
                4. Generate boundary test cases only when meaningful
                   boundaries exist in the source code.
                5. Generate invalid-input cases only when the
                   function explicitly rejects or cannot accept
                   those inputs according to its implementation.
                6. Use exact function names from the source code.
                7. Use actual function arguments.
                8. Never invent function names or arguments.
                9. Do not modify the original source code.
                10. Avoid duplicate test scenarios.
                11. Use realistic input values.
                12. Include zero, negative and mixed-sign values
                    where applicable.
                13. Negative numbers are valid inputs unless the
                    source code explicitly rejects them.
                14. Python integers support arbitrary precision.
                    Do not assume 32-bit or 64-bit integer limits.
                15. Do not invent artificial boundaries.
                16. Consider empty values and unusual inputs
                    when relevant to the function.
                17. Do not calculate expected outputs.
                18. Do not generate assert statements.
                19. Do not generate Python code.
                20. Do not include explanations outside JSON.
                21. Use retrieved testing knowledge only when relevant.
                22. Do not generate tests merely to match retrieved
                    knowledge.

                IMPORTANT:
                Return ONLY valid JSON.
                Do not include Markdown code fences.

                OUTPUT FORMAT:

                Return JSON using exactly this structure:

                {{
                    "test_cases": [
                        {{
                            "description": "Normal test case",
                            "type": "normal",
                            "function": "function_name",
                            "args": [10, 20]
                        }},
                        {{
                            "description": "Edge case with zero",
                            "type": "edge",
                            "function": "function_name",
                            "args": [0, 0]
                        }}
                    ]
                }}

                Allowed test types:
                - normal
                - edge
                - boundary
                - invalid

                Do not add an expected_output field.
                """
            ),
            (
                "human",
                """
                Retrieved Python Testing Knowledge:

                {knowledge}

                --------------------------------

                Python Source Code:

                {code}

                --------------------------------

                Generate accurate test scenarios using the
                relevant retrieved testing knowledge.
                """
            )
        ])

        # Create the LangChain pipeline using LCEL
        self.chain = self.prompt | self.llm

    def generate_tests(self, code):

        # Retrieve relevant testing knowledge from ChromaDB
        retrieved_results = self.retriever.retrieve(
            query="Python testing best practices, edge cases, "
                  "invalid inputs and test scenarios:\n" + code,
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

            knowledge = "No relevant testing knowledge was retrieved."

        # Send code and retrieved knowledge to the LLM
        response = self.chain.invoke({
            "code": code,
            "knowledge": knowledge
        })

        raw_output = response.content.strip()

        # Remove Markdown JSON code fences if present
        raw_output = re.sub(
            r"^```(?:json)?\s*",
            "",
            raw_output,
            flags=re.IGNORECASE
        )

        raw_output = re.sub(
            r"\s*```$",
            "",
            raw_output
        )

        # Parse JSON response
        try:
            data = json.loads(raw_output)

        except json.JSONDecodeError as e:
            raise ValueError(
                f"Test-Writer returned invalid JSON: {e}"
            )

        # Validate JSON structure
        if not isinstance(data, dict):
            raise ValueError(
                "Response must be a JSON object."
            )

        test_cases = data.get("test_cases")

        if not isinstance(test_cases, list) or not test_cases:
            raise ValueError(
                "No valid test scenarios were generated."
            )

        # Allowed test categories
        allowed_types = {
            "normal",
            "edge",
            "boundary",
            "invalid"
        }

        # Validate every test case
        for index, test in enumerate(test_cases):

            if not isinstance(test, dict):
                raise ValueError(
                    f"Test case {index + 1} must be an object."
                )

            if not isinstance(test.get("description"), str):
                raise ValueError(
                    f"Test case {index + 1} has an invalid description."
                )

            if test.get("type") not in allowed_types:
                raise ValueError(
                    f"Test case {index + 1} has an invalid type."
                )

            if not isinstance(test.get("function"), str):
                raise ValueError(
                    f"Test case {index + 1} has an invalid function name."
                )

            if not isinstance(test.get("args"), list):
                raise ValueError(
                    f"Test case {index + 1} must contain an args list."
                )

            # Ensure the LLM does not generate expected outputs
            if "expected_output" in test:
                raise ValueError(
                    "Test-Writer must not generate expected outputs."
                )

        # Return formatted JSON
        return json.dumps(data, indent=4)


# Test the Test-Writer Agent independently
if __name__ == "__main__":

    agent = TestWriterAgent()

    sample_code = """
def calculate_sum_and_double(x, y):
    total_sum = x + y
    doubled_result = total_sum * 2
    return doubled_result
"""

    try:

        print("\n===== GENERATING TESTS WITH RAG =====")

        result = agent.generate_tests(sample_code)

        print("\n===== GENERATED TEST SCENARIOS =====\n")
        print(result)

    except Exception as e:

        print("\nERROR OCCURRED:")
        print(e)
