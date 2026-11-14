
from rag.knowledge_base import KnowledgeBase


class KnowledgeRetriever:

    def __init__(self):

        # Initialize the knowledge base
        self.knowledge_base = KnowledgeBase()

        # Access the ChromaDB collection
        self.collection = self.knowledge_base.collection

    def retrieve(self, query, top_k=3):

        """
        Search the knowledge base and retrieve
        the most relevant documents.
        """

        # Check whether the knowledge base is empty
        if self.collection.count() == 0:

            return []

        # Search ChromaDB using semantic similarity
        results = self.collection.query(
            query_texts=[query],
            n_results=min(
                top_k,
                self.collection.count()
            )
        )

        retrieved_documents = []

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        # Organize retrieved results
        for index, document in enumerate(documents):

            retrieved_documents.append({

                "id": ids[index],

                "category": metadatas[index]["category"],

                "content": document,

                "distance": distances[index]

            })

        return retrieved_documents


# Test the retriever independently
if __name__ == "__main__":

    retriever = KnowledgeRetriever()

    query = "How should I handle division by zero in Python?"

    print("\n===== SEARCH QUERY =====")
    print(query)

    results = retriever.retrieve(query, top_k=3)

    print("\n===== RETRIEVED KNOWLEDGE =====")

    if not results:

        print("No relevant documents found.")

    else:

        for index, result in enumerate(results, start=1):

            print(f"\nDocument {index}")

            print("Category:", result["category"])

            print("Distance:", round(result["distance"], 4))

            print("Content:")
            print(result["content"])

    print("\nRetrieval completed successfully!")
