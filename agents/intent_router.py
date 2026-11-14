
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


class IntentRouter:

    def __init__(self):

        # Initialize local LLM
        self.llm = ChatOllama(
            model="qwen2.5-coder:7b",
            temperature=0
        )

        # Define intent classification prompt
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                You are an intent classification system
                for a Python Code Review Assistant.

                Your task is to identify what the user
                wants to do with their submitted Python code.

                Available intents:

                1. debugging
                   Find logical bugs, runtime errors,
                   and edge-case problems.

                2. style
                   Review naming conventions, formatting,
                   readability, and PEP 8 standards.

                3. fixing
                   Correct problems in the submitted code.

                4. testing
                   Generate test scenarios for the code.

                5. explanation
                   Explain how the submitted code works.

                6. full_review
                   Perform a complete code review using
                   all available agents.

                Classification rules:

                - Identify the user's primary requirement.
                - Do not confuse code fixing with code reviewing.
                - If the user asks to fix code, select fixing.
                - If the user asks to find bugs, select debugging.
                - If the user asks for a complete review,
                  select full_review.
                - If the request is unclear, select full_review.

                IMPORTANT:
                Return ONLY one of these exact values:

                debugging
                style
                fixing
                testing
                explanation
                full_review

                Do not provide explanations.
                Do not use Markdown.
                """
            ),
            (
                "human",
                """
                User Request:

                {request}

                Python Code:

                {code}
                """
            )
        ])

        # Create LCEL chain
        self.chain = self.prompt | self.llm

    def classify(self, request, code=""):

        # Classify the user's request
        response = self.chain.invoke({
            "request": request,
            "code": code
        })

        # Normalize response
        intent = response.content.strip().lower()

        # Allowed intents
        allowed_intents = {
            "debugging",
            "style",
            "fixing",
            "testing",
            "explanation",
            "full_review"
        }

        # Validate the LLM response
        if intent not in allowed_intents:

            return "full_review"

        return intent


# Test Intent Router independently
if __name__ == "__main__":

    router = IntentRouter()

    sample_code = """
def calculate_sum(a, b):
    return a + b
"""

    test_requests = [
        "Find bugs in my code",
        "Check my coding style",
        "Fix my Python code",
        "Generate test cases",
        "Explain how this code works",
        "Perform a complete code review"
    ]

    print("\n===== INTENT ROUTER TEST =====")

    for request in test_requests:

        intent = router.classify(
            request=request,
            code=sample_code
        )

        print("\nUser Request:", request)
        print("Detected Intent:", intent)
