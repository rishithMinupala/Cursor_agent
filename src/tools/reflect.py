import time

from langchain_core.messages import SystemMessage, HumanMessage

from src.config import (
    MODEL_MAX_RETRIES,
    MODEL_RETRY_BACKOFF_FACTOR,
    MODEL_RETRY_INITIAL_DELAY,
    REFLECT_MAX_OUTPUT_CHARS,
)

REFLECT_PROMPT = """You are reviewing your own code changes. Given the list of files changed and a short summary, produce a brief reflection: what was done, whether it matches the task, and any obvious risks or follow-ups. Keep under 300 words."""

CRITIQUE_PROMPT = """You are a code reviewer. Given the list of files changed and a short summary, produce a brief critique only for: correctness, bugs, logic errors, and edge cases that could break behavior. Do not suggest adding comments, docstrings, or style-only changes. Only suggest changes that fix real issues or meaningfully improve/optimize behavior. If the code is correct and complete for the task, say so briefly and do not ask for non-essential edits. Keep under 200 words."""


def _is_transient(e: Exception) -> bool:
    if isinstance(e, (ConnectionError, TimeoutError, OSError)):
        return True
    name = type(e).__name__
    return "ProtocolError" in name or "RemoteProtocol" in name or "ConnectError" in name or "ReadTimeout" in name


def _call_llm(model, messages) -> str:
    if model is None:
        return "Error: no model available for reflection/critique."
    delay = MODEL_RETRY_INITIAL_DELAY
    last_exc = None
    for attempt in range(MODEL_MAX_RETRIES + 1):
        try:
            out = model.invoke(messages)
            text = getattr(out, "content", str(out)) or ""
            return text[:REFLECT_MAX_OUTPUT_CHARS]
        except Exception as e:
            last_exc = e
            if not _is_transient(e) or attempt >= MODEL_MAX_RETRIES:
                return f"Error: {e}"
            time.sleep(min(delay, 60))
            delay *= MODEL_RETRY_BACKOFF_FACTOR
    return f"Error: {last_exc}"


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
