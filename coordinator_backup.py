
import ast
import json
import os
import re
import subprocess
import sys
import tempfile

from agents.bug_finder import BugFinderAgent
from agents.style_lint import StyleLintAgent
from agents.fixer import FixerAgent
from agents.test_writer import TestWriterAgent

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

        # Initialize Expected-Result Validator
        self.validator = ExpectedResultValidator()

        print("All agents initialized successfully!")

    def execute_tests(self, fixed_code, test_cases):
        """
        Execute generated test scenarios against fixed code.

        This performs smoke testing and records:
        - Function execution status
        - Returned values
        - Runtime errors

        Logical correctness is checked separately
        by the Expected-Result Validator.
        """

        fixed_code = clean_code(fixed_code)

        # Parse generated test scenarios
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

        # Identify actual functions in the fixed code
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

        # Validate generated scenarios
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
                        "in the fixed code."
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

            # Create temporary Python file
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                delete=False,
                encoding="utf-8"
            ) as temp_file:

                temp_file.write(harness_code)
                temp_path = temp_file.name

            # Execute using current Python interpreter
            process = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=15
            )

            # Extract structured results
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

            # Remove temporary file
            if temp_path and os.path.exists(temp_path):

                os.remove(temp_path)

    def review_code(self, code):

        print("\n" + "=" * 60)
        print("STARTING CODE REVIEW")
        print("=" * 60)

        # 1. Bug-Finder Agent
        print("\n[1/6] Running Bug-Finder Agent...")

        bug_report = self.bug_finder.analyze(code)

        # 2. Style/Lint Agent
        print("\n[2/6] Running Style/Lint Agent...")

        style_report = self.style_lint.review(code)

        # 3. Fixer Agent
        print("\n[3/6] Running Fixer Agent...")

        fixed_code = self.fixer.fix(
            code,
            bug_report + "\n" + style_report
        )

        fixed_code = clean_code(fixed_code)

        # 4. Test-Writer Agent
        print("\n[4/6] Running Test-Writer Agent...")

        test_cases = self.test_writer.generate_tests(fixed_code)

        # 5. Test Execution
        print("\n[5/6] Executing generated test scenarios...")

        test_result = self.execute_tests(
            fixed_code,
            test_cases
        )

        # 6. Expected-Result Validation
        print("\n[6/6] Validating expected results...")

        validation_report = self.validator.validate(
            test_cases,
            test_result
        )

        print("\nCODE REVIEW COMPLETED!")

        return {
            "bug_report": bug_report,
            "style_report": style_report,
            "fixed_code": fixed_code,
            "test_cases": test_cases,
            "test_result": test_result,
            "validation_report": validation_report
        }


# Run the complete multi-agent system
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

        report = coordinator.review_code(sample_code)

        print("\n" + "=" * 60)
        print("FINAL CODE REVIEW REPORT")
        print("=" * 60)

        print("\n===== BUG REPORT =====")
        print(report["bug_report"])

        print("\n===== STYLE REPORT =====")
        print(report["style_report"])

        print("\n===== FIXED CODE =====")
        print(report["fixed_code"])

        print("\n===== GENERATED TEST SCENARIOS =====")
        print(report["test_cases"])

        print("\n===== TEST EXECUTION REPORT =====")

        print(json.dumps(
            report["test_result"],
            indent=4
        ))

        print("\n===== EXPECTED-RESULT VALIDATION REPORT =====")

        print(json.dumps(
            report["validation_report"],
            indent=4
        ))

    except Exception as e:

        print("\nERROR DURING CODE REVIEW:")
        print(e)
