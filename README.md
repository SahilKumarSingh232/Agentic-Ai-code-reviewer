# Code Review & Debug Squad

A Streamlit code assistant that coordinates focused Python agents for bug
finding, style review, code fixing, test generation, test execution, and code
explanation. The agents use a small ChromaDB knowledge base to retrieve Python
guidance before responding.

## Requirements

- Python 3.10 or newer
- [Ollama](https://ollama.com/) installed and running
- The `qwen2.5-coder:7b` model pulled locally

## Run locally

From this project folder, create and activate a virtual environment, then run:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
ollama pull qwen2.5-coder:7b
ollama serve
```

In a second terminal with the environment activated, initialize the local
knowledge base and start the app:

```powershell
python -c "from rag.knowledge_base import KnowledgeBase; KnowledgeBase().add_documents()"
streamlit run app.py
```

The app opens in your browser. Paste Python code, choose the kind of review,
and select **Run review**. `coordinator.py` can also be run directly for a
terminal demonstration.

## Project layout

- `app.py` - Streamlit interface
- `coordinator.py` - intent routing and agent orchestration
- `agents/` - bug finder, style/lint, fixer, test writer, and intent router
- `rag/` - local ChromaDB knowledge base and retriever
- `utils/expected_result_validator.py` - compares results for functions with
  registered reference implementations
- `data/chroma_db/` - persisted local vector database

## Execution note

Generated test calls run in a separate Python process with a time limit, but
that process is not isolated by an operating-system security sandbox. Review
code before running it and do not submit untrusted code on a machine with
sensitive files or credentials. Expected-result checks are available only for
functions with a trusted reference implementation registered in the validator;
other function results are reported as unverified.
