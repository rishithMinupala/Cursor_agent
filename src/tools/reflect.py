from langchain_core.messages import SystemMessage, HumanMessage

from src.config import REFLECT_MAX_OUTPUT_CHARS

REFLECT_PROMPT = """You are reviewing your own code changes. Given the list of files changed and a short summary, produce a brief reflection: what was done, whether it matches the task, and any obvious risks or follow-ups. Keep under 300 words."""

CRITIQUE_PROMPT = """You are a code reviewer. Given the list of files changed and a short summary of the change, produce a brief critique: correctness, style, edge cases, and suggestions. Keep under 300 words."""


def _call_llm(model, messages) -> str:
    if model is None:
        return "Error: no model available for reflection/critique."
    try:
        out = model.invoke(messages)
        text = getattr(out, "content", str(out)) or ""
        return text[:REFLECT_MAX_OUTPUT_CHARS]
    except Exception as e:
        return f"Error: {e}"


def reflect_on_changes_impl(state: dict, args: dict, model=None) -> str:
    files = args.get("files_changed") or state.get("files_changed") or []
    summary = args.get("summary") or state.get("working_summary") or "No summary."
    content = f"Files changed: {files}\n\nSummary: {summary}"
    messages = [SystemMessage(content=REFLECT_PROMPT), HumanMessage(content=content)]
    return _call_llm(model, messages)


def critique_changes_impl(state: dict, args: dict, model=None) -> str:
    files = args.get("files_changed") or state.get("files_changed") or []
    summary = args.get("summary") or state.get("working_summary") or "No summary."
    content = f"Files changed: {files}\n\nSummary: {summary}"
    messages = [SystemMessage(content=CRITIQUE_PROMPT), HumanMessage(content=content)]
    return _call_llm(model, messages)
