"""Streamlit interface for the Code Review & Debug Squad."""

import json

import streamlit as st


st.set_page_config(
    page_title="Code Review & Debug Squad",
    page_icon="💻",
    layout="wide",
)

st.title("Code Review & Debug Squad")
st.caption("A small AI engineering team for Python reviews, fixes, and tests.")

with st.sidebar:
    st.header("Review options")
    request = st.selectbox(
        "What should the squad do?",
        [
            "Perform a complete code review",
            "Find bugs and edge cases",
            "Review style and readability",
            "Fix the code and verify it",
            "Generate and run tests",
            "Explain the code",
        ],
    )
    st.info("The project uses the local Ollama model `qwen2.5-coder:7b`.")

default_code = '''def calculate_sum_and_double(first_number, second_number):
    total = first_number + second_number
    return total * 2
'''
code = st.text_area(
    "Python code",
    value=default_code,
    height=320,
    placeholder="Paste your Python code here...",
)

if st.button("Run review", type="primary", disabled=not code.strip()):
    try:
        with st.spinner("The agents are reviewing your code..."):
            from coordinator import CodeReviewCoordinator

            coordinator = CodeReviewCoordinator()
            report = coordinator.review_code(code=code, request=request)
        st.session_state["review_report"] = report
    except Exception as error:
        st.error(f"Review could not be completed: {error}")
        st.caption(
            "Check that Ollama is running, the qwen2.5-coder:7b model is "
            "available, and the project dependencies are installed."
        )

report = st.session_state.get("review_report")
if report:
    st.divider()
    st.subheader("Consolidated review")
    st.markdown(f"**Detected request:** `{report['intent']}`")

    sections = [
        ("Bug findings", "bug_report"),
        ("Style and lint review", "style_report"),
        ("Code explanation", "explanation"),
    ]
    for title, key in sections:
        if report.get(key):
            with st.expander(title, expanded=True):
                st.markdown(report[key])

    if report.get("fixed_code"):
        st.subheader("Suggested corrected code")
        st.code(report["fixed_code"], language="python")
        st.download_button(
            "Download corrected code",
            report["fixed_code"],
            file_name="reviewed_code.py",
            mime="text/x-python",
        )

    if report.get("test_cases"):
        with st.expander("Generated test scenarios", expanded=True):
            try:
                st.json(json.loads(report["test_cases"]))
            except (TypeError, json.JSONDecodeError):
                st.code(report["test_cases"], language="json")

    if report.get("test_result"):
        with st.expander("Test execution", expanded=True):
            st.json(report["test_result"])
            st.caption(
                "Execution results show whether calls completed. "
                "Expected-result validation is shown separately."
            )

    if report.get("validation_report"):
        with st.expander("Expected-result validation", expanded=True):
            st.json(report["validation_report"])

st.caption(
    "Only run code you trust. The execution helper uses a timeout but is "
    "not an operating-system security sandbox."
)
