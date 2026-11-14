
class ExpectedResultValidator:
    """
    Validates actual function outputs against independently
    calculated expected outputs from trusted reference functions.
    """

    def __init__(self):

        # Trusted reference implementations
        self.reference_functions = {}

        # Register reference implementations for our demo functions
        self.register_reference(
            "calculate_sum",
            self.calculate_sum_reference
        )

        self.register_reference(
            "calculate_sum_and_double",
            self.calculate_sum_reference
        )

    @staticmethod
    def calculate_sum_reference(first_number, second_number):
        """
        Independent reference implementation.

        Expected behavior:
        Add two numbers and multiply the result by 2.
        """

        return (first_number + second_number) * 2

    def register_reference(self, function_name, reference_function):
        """
        Register a trusted reference implementation.
        """

        self.reference_functions[function_name] = reference_function

    def validate(self, test_cases, execution_report):
        """
        Compare actual execution results with expected results.
        """

        # Convert JSON string into dictionary if necessary
        if isinstance(test_cases, str):
            import json
            test_cases = json.loads(test_cases)

        scenarios = test_cases.get("test_cases", [])

        actual_results = execution_report.get("results", [])

        # Match execution results using test numbers
        actual_by_number = {
            result["test_number"]: result
            for result in actual_results
        }

        validation_results = []

        for index, test in enumerate(scenarios, start=1):

            function_name = test["function"]
            arguments = test["args"]

            actual = actual_by_number.get(index)

            result = {
                "test_number": index,
                "description": test["description"],
                "function": function_name,
                "arguments": arguments
            }

            # Check whether execution result exists
            if actual is None:

                result["status"] = "EXECUTION_ERROR"
                result["details"] = "No execution result was found."

                validation_results.append(result)
                continue

            # Check whether the function executed successfully
            if actual["status"] != "COMPLETED":

                result["status"] = "EXECUTION_ERROR"
                result["details"] = actual.get(
                    "error",
                    "Function execution failed."
                )

                validation_results.append(result)
                continue

            # Check whether a trusted reference exists
            reference_function = self.reference_functions.get(
                function_name
            )

            if reference_function is None:

                result["status"] = "UNVERIFIED"
                result["details"] = (
                    "No trusted reference implementation "
                    "is registered for this function."
                )

                validation_results.append(result)
                continue

            try:

                # Calculate expected output independently
                expected_output = reference_function(*arguments)

                # Get actual output from execution report
                actual_output = actual["returned_value"]

                # Compare expected and actual values
                if actual_output == repr(expected_output):

                    result["status"] = "PASS"

                else:

                    result["status"] = "FAIL"

                result["expected_output"] = repr(expected_output)
                result["actual_output"] = actual_output

            except Exception as error:

                result["status"] = "VALIDATION_ERROR"
                result["details"] = str(error)

            validation_results.append(result)

        # Calculate summary
        passed = sum(
            1 for item in validation_results
            if item["status"] == "PASS"
        )

        failed = sum(
            1 for item in validation_results
            if item["status"] == "FAIL"
        )

        unverified = sum(
            1 for item in validation_results
            if item["status"] == "UNVERIFIED"
        )

        execution_errors = sum(
            1 for item in validation_results
            if item["status"] in {
                "EXECUTION_ERROR",
                "VALIDATION_ERROR"
            }
        )

        if failed > 0 or execution_errors > 0:
            overall_status = "ISSUES FOUND"

        elif unverified > 0:
            overall_status = "PARTIALLY VERIFIED"

        else:
            overall_status = "ALL TESTS VERIFIED"

        return {
            "overall_status": overall_status,
            "total_tests": len(validation_results),
            "passed": passed,
            "failed": failed,
            "unverified": unverified,
            "execution_errors": execution_errors,
            "results": validation_results
        }


# Test the Expected-Result Validator independently
if __name__ == "__main__":

    import json

    validator = ExpectedResultValidator()

    # Sample test scenarios
    test_cases = {
        "test_cases": [
            {
                "description": "Normal positive inputs",
                "type": "normal",
                "function": "calculate_sum",
                "args": [10, 20]
            },
            {
                "description": "Both inputs are zero",
                "type": "edge",
                "function": "calculate_sum",
                "args": [0, 0]
            },
            {
                "description": "Negative inputs",
                "type": "edge",
                "function": "calculate_sum",
                "args": [-10, -20]
            },
            {
                "description": "Unknown function",
                "type": "normal",
                "function": "unknown_function",
                "args": [5, 10]
            }
        ]
    }

    # Simulated output from the Test Execution Engine
    execution_report = {
        "results": [
            {
                "test_number": 1,
                "status": "COMPLETED",
                "returned_value": "60"
            },
            {
                "test_number": 2,
                "status": "COMPLETED",
                "returned_value": "0"
            },
            {
                "test_number": 3,
                "status": "COMPLETED",
                "returned_value": "-60"
            },
            {
                "test_number": 4,
                "status": "COMPLETED",
                "returned_value": "15"
            }
        ]
    }

    # Validate results
    report = validator.validate(
        test_cases,
        execution_report
    )

    print("\n===== EXPECTED-RESULT VALIDATION REPORT =====\n")

    print(json.dumps(report, indent=4))
