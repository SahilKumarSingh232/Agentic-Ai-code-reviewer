
import ast
import json
import os
import re
import subprocess
import sys
import tempfile

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from agents.bug_finder import BugFinderAgent
from agents.style_lint import StyleLintAgent
from agents.fixer import FixerAgent
from agents.test_writer import TestWriterAgent
from agents.intent_router import IntentRouter

from utils.expected_result_validator import ExpectedResultValidator


def clean_code(code):
    """Remove Markdown code fences from generated code."""

    code = re.sub(
        r"```(?:python)?\s*",
        "",
        code,
        flags=re.IGNORECASE
    )

    code = code.replace("```", "")

    return code.strip()


class CodeReviewCoordinator:

    def __init__(self):

        print("Initializing Code Review Agents...")

        self.bug_finder = BugFinderAgent()
        self.style_lint = StyleLintAgent()
        self.fixer = FixerAgent()
        self.test_writer = TestWriterAgent()
        self.intent_router = IntentRouter()

        self.validator = ExpectedResultValidator()

        # Explanation LLM
        self.explanation_llm = ChatOllama(
            model="qwen2.5-coder:7b",
            temperature=0
        )

        # Explanation prompt
        self.explanation_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                You are an expert Python programming teacher.

                Explain the submitted Python code in simple,
                beginner-friendly language.

                Include:
                1. Purpose of the code.
                2. Step-by-step working.
                3. Explanation of important functions and variables.
                4. Expected output, if it can be determined.
                5. Any important observations.

                Do not modify the submitted code.
                """
            ),
            (
                "human",
                """
                Explain the following Python code:

                {code}
                """
            )
        ])

        # LCEL explanation chain
        self.explanation_chain = (
            self.explanation_prompt | self.explanation_llm
        )

        print("All agents initialized successfully!")

    def execute_tests(self, fixed_code, test_cases):
        """
        Execute generated test scenarios.

        This checks whether functions execute successfully
        or produce runtime errors.

        It does not independently prove logical correctness.
        """

        fixed_code = clean_code(fixed_code)

        try:
            test_data = json.loads(test_cases)

        except json.JSONDecodeError as e:

            return {
                "status": "INVALID TEST DATA",
                "details": str(e)
            }

        scenarios = test_data.get("test_cases")

        if not isinstance(scenarios, list) or not scenarios:

            return {
                "status": "NO TEST CASES",
                "details": "No test scenarios were provided."
            }

        try:
            syntax_tree = ast.parse(fixed_code)

        except SyntaxError as e:

            return {
                "status": "INVALID FIXED CODE",
                "details": str(e)
            }

        available_functions = {
            node.name
            for node in ast.walk(syntax_tree)
            if isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef)
            )
        }

        # Validate test scenarios
        for index, test in enumerate(scenarios):

            if not isinstance(test, dict):

                return {
                    "status": "INVALID TEST DATA",
                    "details": f"Test case {index + 1} is not an object."
                }

            function_name = test.get("function")

            if function_name not in available_functions:

                return {
                    "status": "INVALID TEST DATA",
                    "details": (
                        f"Function '{function_name}' was not found "
                        "in the submitted code."
                    )
                }

            if not isinstance(test.get("args"), list):

                return {
                    "status": "INVALID TEST DATA",
                    "details": (
                        f"Test case {index + 1} has invalid arguments."
                    )
                }

        # Prepare execution script
        source_literal = repr(fixed_code)
        tests_literal = repr(json.dumps(scenarios))

        harness_code = "\n".join([

            "import json",
            "import io",
            "import contextlib",
            "",

            f"SOURCE_CODE = {source_literal}",
            f"TEST_CASES = json.loads({tests_literal})",

            "",
            "namespace = {}",
            "results = []",
            "captured_output = io.StringIO()",

            "# Load submitted source code",
            "try:",
            "    with contextlib.redirect_stdout(captured_output):",
            "        exec(compile(SOURCE_CODE, '<reviewed_code>', 'exec'), namespace)",
            "except BaseException as error:",
            "    print('__RESULT__' + json.dumps({",
            "        'load_error': str(error)",
            "    }))",
            "    raise SystemExit(0)",

            "",
            "# Execute generated test scenarios",
            "for index, test in enumerate(TEST_CASES, start=1):",

            "    function_name = test['function']",
            "    arguments = test['args']",

            "    try:",

            "        function = namespace[function_name]",

            "        with contextlib.redirect_stdout(captured_output):",
            "            result = function(*arguments)",

            "        results.append({",
            "            'test_number': index,",
            "            'description': test['description'],",
            "            'type': test['type'],",
            "            'function': function_name,",
            "            'arguments': arguments,",
            "            'status': 'COMPLETED',",
            "            'returned_value': repr(result)",
            "        })",

            "    except BaseException as error:",

            "        results.append({",
            "            'test_number': index,",
            "            'description': test['description'],",
            "            'type': test['type'],",
            "            'function': function_name,",
            "            'arguments': arguments,",
            "            'status': 'ERROR',",
            "            'error': str(error)",
            "        })",

            "",
            "print('__RESULT__' + json.dumps(results))"
        ])

        temp_path = None

        try:

            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                delete=False,
                encoding="utf-8"
            ) as temp_file:

                temp_file.write(harness_code)
                temp_path = temp_file.name

            process = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=15
            )

            result_line = None

            for line in process.stdout.splitlines():

                if line.startswith("__RESULT__"):

                    result_line = line[len("__RESULT__"):]

            if result_line is None:

                return {
                    "status": "EXECUTION FAILED",
                    "details": process.stderr or process.stdout
                }

            parsed_results = json.loads(result_line)

            if (
                isinstance(parsed_results, dict)
                and "load_error" in parsed_results
            ):

                return {
                    "status": "CODE EXECUTION FAILED",
                    "details": parsed_results["load_error"]
                }

            completed = sum(
                1 for result in parsed_results
                if result["status"] == "COMPLETED"
            )

            errors = sum(
                1 for result in parsed_results
                if result["status"] == "ERROR"
            )

            overall_status = (
                "COMPLETED" if errors == 0 else "ERRORS FOUND"
            )

            return {
                "status": overall_status,
                "total_tests": len(parsed_results),
                "completed": completed,
                "errors": errors,
                "results": parsed_results,
                "note": (
                    "COMPLETED means the function ran without raising "
                    "an exception. It does not prove the returned value "
                    "is logically correct."
                )
            }

        except subprocess.TimeoutExpired:

            return {
                "status": "TIMEOUT",
                "details": "Test execution exceeded 15 seconds."
            }

        except Exception as e:

            return {
                "status": "EXECUTION FAILED",
                "details": str(e)
            }

        finally:

            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    def generate_and_execute_tests(self, code):

        print("\nRunning Test-Writer Agent...")

        test_cases = self.test_writer.generate_tests(code)

        print("\nExecuting generated test scenarios...")

        test_result = self.execute_tests(
            code,
            test_cases
        )

        print("\nValidating expected results...")

        validation_report = self.validator.validate(
            test_cases,
            test_result
        )

        return test_cases, test_result, validation_report

    def review_code(self, code, request="Perform a complete code review"):

        print("\n" + "=" * 60)
        print("STARTING CODE REVIEW")
        print("=" * 60)

        # Step 1: Identify user intent
        print("\nIdentifying user intent...")

        intent = self.intent_router.classify(
            request=request,
            code=code
        )

        print("Detected Intent:", intent)

        # Common report structure
        report = {
            "intent": intent,
            "bug_report": None,
            "style_report": None,
            "fixed_code": None,
            "test_cases": None,
            "test_result": None,
            "validation_report": None,
            "explanation": None
        }

        # Route 1: Debugging
        if intent == "debugging":

            print("\nRunning Bug-Finder Agent...")

            report["bug_report"] = self.bug_finder.analyze(code)

        # Route 2: Style review
        elif intent == "style":

            print("\nRunning Style/Lint Agent...")

            report["style_report"] = self.style_lint.review(code)

        # Route 3: Code fixing
        elif intent == "fixing":

            print("\nRunning Bug-Finder Agent...")

            bug_report = self.bug_finder.analyze(code)

            print("\nRunning Style/Lint Agent...")

            style_report = self.style_lint.review(code)

            print("\nRunning Fixer Agent...")

            fixed_code = self.fixer.fix(
                code,
                bug_report + "\n" + style_report
            )

            fixed_code = clean_code(fixed_code)

            report["bug_report"] = bug_report
            report["style_report"] = style_report
            report["fixed_code"] = fixed_code

            (
                report["test_cases"],
                report["test_result"],
                report["validation_report"]
            ) = self.generate_and_execute_tests(fixed_code)

        # Route 4: Testing
        elif intent == "testing":

            (
                report["test_cases"],
                report["test_result"],
                report["validation_report"]
            ) = self.generate_and_execute_tests(code)

        # Route 5: Explanation
        elif intent == "explanation":

            print("\nGenerating code explanation...")

            response = self.explanation_chain.invoke({
                "code": code
            })

            report["explanation"] = response.content

        # Route 6: Complete code review
        elif intent == "full_review":

            print("\n[1/6] Running Bug-Finder Agent...")

            bug_report = self.bug_finder.analyze(code)

            print("\n[2/6] Running Style/Lint Agent...")

            style_report = self.style_lint.review(code)

            print("\n[3/6] Running Fixer Agent...")

            fixed_code = self.fixer.fix(
                code,
                bug_report + "\n" + style_report
            )

            fixed_code = clean_code(fixed_code)

            print("\n[4/6] Running Test-Writer Agent...")

            test_cases = self.test_writer.generate_tests(fixed_code)

            print("\n[5/6] Executing generated test scenarios...")

            test_result = self.execute_tests(
                fixed_code,
                test_cases
            )

            print("\n[6/6] Validating expected results...")

            validation_report = self.validator.validate(
                test_cases,
                test_result
            )

            report["bug_report"] = bug_report
            report["style_report"] = style_report
            report["fixed_code"] = fixed_code
            report["test_cases"] = test_cases
            report["test_result"] = test_result
            report["validation_report"] = validation_report

        print("\nREQUEST COMPLETED!")

        return report


# Run the coordinator independently
if __name__ == "__main__":

    coordinator = CodeReviewCoordinator()

    sample_code = """
def CALCULATE(x,y):
    a=x+y
    b=a*2
    return b

print(CALCULATE(10,20))
"""

    try:

        report = coordinator.review_code(
            code=sample_code,
            request="Perform a complete code review"
        )

        print("\n" + "=" * 60)
        print("FINAL CODE REVIEW REPORT")
        print("=" * 60)

        print("\n===== DETECTED INTENT =====")
        print(report["intent"])

        if report["bug_report"]:
            print("\n===== BUG REPORT =====")
            print(report["bug_report"])

        if report["style_report"]:
            print("\n===== STYLE REPORT =====")
            print(report["style_report"])

        if report["fixed_code"]:
            print("\n===== FIXED CODE =====")
            print(report["fixed_code"])

        if report["test_cases"]:
            print("\n===== GENERATED TEST SCENARIOS =====")
            print(report["test_cases"])

        if report["test_result"]:
            print("\n===== TEST EXECUTION REPORT =====")
            print(json.dumps(
                report["test_result"],
                indent=4
            ))

        if report["validation_report"]:
            print("\n===== EXPECTED-RESULT VALIDATION REPORT =====")
            print(json.dumps(
                report["validation_report"],
                indent=4
            ))

        if report["explanation"]:
            print("\n===== CODE EXPLANATION =====")
            print(report["explanation"])

    except Exception as e:

        print("\nERROR DURING CODE REVIEW:")
        print(e)
