
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction
)


class KnowledgeBase:

    def __init__(self):

        # Store the database inside the project
        project_root = Path(__file__).resolve().parent.parent

        database_path = project_root / "data" / "chroma_db"

        database_path.mkdir(parents=True, exist_ok=True)

        # Initialize persistent ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(database_path)
        )

        # Initialize local embedding model
        self.embedding_function = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2",
            device="cpu"
        )

        # Create or access the collection
        self.collection = self.client.get_or_create_collection(
            name="python_knowledge",
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self):

        # Python programming knowledge documents
        documents = [

            {
                "id": "python_001",
                "category": "naming",
                "content": """
                Python Naming Conventions:

                Use lowercase letters with underscores for function
                and variable names. This is known as snake_case.

                Function names should describe what the function does.

                Avoid unclear names such as a, b, x, and temp when
                meaningful names can be used.

                Constants should generally use uppercase letters
                with underscores.

                Example:
                def calculate_total(first_number, second_number):
                    total = first_number + second_number
                    return total
                """
            },

            {
                "id": "python_002",
                "category": "formatting",
                "content": """
                Python Code Formatting:

                Use four spaces for indentation.

                Add spaces around operators such as +, -, *, and =.

                Use a space after commas.

                Keep code readable and avoid unnecessarily long lines.

                Follow the PEP 8 Python style guidelines.
                """
            },

            {
                "id": "python_003",
                "category": "functions",
                "content": """
                Python Function Best Practices:

                A function should perform a clear and specific task.

                Use descriptive function names.

                Avoid unnecessarily large functions.

                Use parameters to pass input values.

                Return results when the caller needs to use them.

                Add a docstring explaining the function's purpose,
                parameters, and return value.
                """
            },

            {
                "id": "python_004",
                "category": "exceptions",
                "content": """
                Python Exception Handling:

                Use try and except blocks to handle expected errors.

                Catch specific exceptions instead of using a broad
                except clause whenever possible.

                ZeroDivisionError occurs when a number is divided by zero.

                ValueError occurs when a function receives an
                argument with an inappropriate value.

                TypeError occurs when an operation is applied to
                an incompatible data type.

                Use finally when cleanup must happen regardless
                of whether an exception occurs.
                """
            },

            {
                "id": "python_005",
                "category": "logic_errors",
                "content": """
                Common Python Logical Errors:

                Logical errors occur when a program executes but
                produces an incorrect result.

                Common examples include incorrect mathematical
                operators, wrong conditions, incorrect loop limits,
                and incorrect variable usage.

                Always check whether calculations and conditions
                match the intended requirements.

                Test normal cases and edge cases to detect errors.
                """
            },

            {
                "id": "python_006",
                "category": "edge_cases",
                "content": """
                Python Edge Case Testing:

                Edge cases are unusual or extreme inputs that may
                expose problems in a program.

                Examples include zero, negative numbers, empty
                strings, empty lists, and very large numbers.

                The correct edge cases depend on the function's
                requirements and accepted input types.

                Python integers support arbitrary precision, so
                they are not restricted to 32-bit integer limits.
                """
            },

            {
                "id": "python_007",
                "category": "readability",
                "content": """
                Python Code Readability:

                Write code that is easy for other developers to
                understand and maintain.

                Use meaningful variable names.

                Avoid unnecessary duplicate code.

                Keep related operations together.

                Add comments when they explain why something is
                done, rather than repeating what the code says.
                """
            },

            {
                "id": "python_008",
                "category": "testing",
                "content": """
                Python Testing Best Practices:

                Test functions with normal inputs, edge cases,
                and invalid inputs where applicable.

                Check the returned values against expected results.

                Verify that invalid inputs produce appropriate
                errors when the function is expected to reject them.

                Test each function independently when possible.

                A function executing without an exception does
                not necessarily mean its output is correct.
                """
            }
        ]

        # Extract document information
        ids = [doc["id"] for doc in documents]

        contents = [doc["content"] for doc in documents]

        metadatas = [
            {"category": doc["category"]}
            for doc in documents
        ]

        # Store documents and generate embeddings
        self.collection.upsert(
            ids=ids,
            documents=contents,
            metadatas=metadatas
        )

        print(f"{len(documents)} knowledge documents processed.")

    def get_document_count(self):

        return self.collection.count()


# Run the knowledge base independently
if __name__ == "__main__":

    print("Initializing Python Knowledge Base...")

    knowledge_base = KnowledgeBase()

    knowledge_base.add_documents()

    print(
        "Total documents in ChromaDB:",
        knowledge_base.get_document_count()
    )

    print("Knowledge Base initialized successfully!")
