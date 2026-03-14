import logging
import subprocess
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Overwrite
from pydantic import BaseModel, Field

from src.config import (
    MAX_TOOL_RESULT_CHARS,
    MODEL_MAX_RETRIES,
    MODEL_RETRY_BACKOFF_FACTOR,
    MODEL_RETRY_INITIAL_DELAY,
    RECENT_MESSAGES_KEEP,
    SUMMARY_TRIGGER_MESSAGES,
    TEST_ENTRY_POINT_TIMEOUT,
    TEST_OUTPUT_MAX_CHARS,
    TEST_PYTEST_TIMEOUT,
    TOOL_MAX_RETRIES,
    TOOL_RETRY_BACKOFF_FACTOR,
    TOOL_RETRY_EXCEPTIONS,
    TOOL_RETRY_INITIAL_DELAY,
)
from src.state import AgentState
from src.tools import TOOL_IMPLS
from src.tools.reflect import critique_changes_impl
from src.tools.schemas import CODER_TOOLS, GIT_TOOLS

logger = logging.getLogger(__name__)


class TestPlan(BaseModel):
    run_pytest: bool = Field(description="True if pytest should be run (repo has tests and code was changed).")
    run_entry_point: bool = Field(description="True if the app should be run from its entry point to check for startup/runtime errors.")
    entry_point_command: str | None = Field(default=None, description="Shell command to run the app, e.g. 'python -m src.main' or 'python main.py'. Required when run_entry_point is True.")
    reason: str | None = Field(default=None, description="Brief reason for this plan.")

CODER_PROMPT = """You are a coding agent. Implement the user's feature or bugfix.

Workflow:
1. pull_repo if needed, then grep_search/read_code only until you know what to change. Do not loop: use each at most once or twice per file.
2. Apply the fix with write_code or edit_in_file (replace/insert_after/insert_before). Lines are 1-based.
3. CRITICAL: As soon as you have applied the fix (write_code or edit_in_file), do NOT call grep_search or read_code again. Your next step must be: call critique_changes once, then respond with one short message (e.g. "Done with code changes.") and no tool calls. The workflow will run the code and open a PR.
4. If critique_changes suggests a real fix (bug/correctness), make that edit and call critique_changes again; then respond "Done with code changes." with no tool calls. Do not re-read or re-grep after the edit.
5. Do not add test files or test directories. Do not read requirements.txt or other optional files. Do not call git tools."""

GIT_PROMPT = """You are a git/PR agent. You MUST use tool calls only; do not respond with plain text or a summary.

Call these tools in order (you may call multiple in one response):
1. create_branch – branch_name from the task (e.g. bugfix/calc-add, feature/xyz).
2. commit_changes – message: short commit message from the task/summary.
3. push – no args.
4. create_pr – title and body from the task context.

Your first response must include at least one tool call (start with create_branch). Do not output a summary or title as text; use the tools."""

TESTER_SYSTEM = """You are a testing agent. Given the conversation (user task, what the coder did) and the list of changed files, decide how to verify the changes.

Rules:
- If ONLY docs/config were changed (e.g. only .md, .txt, .yml, .yaml, .json, .toml, .env): set run_pytest=false and run_entry_point=false. Critique was already done; no automated test needed.
- If application code was changed (.py, .js, .ts, etc.): set run_entry_point=true and provide the shell command to run the app from its entry point. For Python repos, prefer "python main.py" when main.py exists at repo root; otherwise "python -m src.main" or similar. Do not use "python src/foo.py" as entry point unless that file is the app entry. Use "python main.py" or "node index.js" etc. runnable from repo root.
- Set run_pytest=true only if the repo clearly has a test suite (e.g. tests/ directory with test files). If unsure or minimal repo, set run_pytest=false.
- entry_point_command must be a single command runnable from the repo root. No shell operators like && or |."""


def _is_transient_model_error(e: Exception) -> bool:
    if isinstance(e, (ConnectionError, TimeoutError, OSError)):
        return True
    name = type(e).__name__
    return "ProtocolError" in name or "RemoteProtocol" in name or "ConnectError" in name or "ReadTimeout" in name

def create_graph(llm, checkpointer=None):
    coder_model = llm.bind_tools(CODER_TOOLS)
    git_model = llm.bind_tools(GIT_TOOLS)
    tool_impls = dict(TOOL_IMPLS)
    tool_impls["critique_changes"] = lambda s, a: critique_changes_impl(s, a, llm)

    def invoke_with_retry(model, input, config=None):
        delay = MODEL_RETRY_INITIAL_DELAY
        last_exc = None
        for attempt in range(MODEL_MAX_RETRIES + 1):
            try:
                if config is not None:
                    return model.invoke(input, config)
                return model.invoke(input)
            except Exception as e:
                last_exc = e
                if not _is_transient_model_error(e) or attempt >= MODEL_MAX_RETRIES:
                    raise
                logger.info("model retry after %s (attempt %s)", type(e).__name__, attempt + 1)
                time.sleep(min(delay, 60))
                delay *= MODEL_RETRY_BACKOFF_FACTOR
        if last_exc is not None:
            raise last_exc

    def summarize_node(state: AgentState) -> dict:
        msgs = state.get("messages") or []
        if len(msgs) <= SUMMARY_TRIGGER_MESSAGES:
            return {}
        logger.info("phase=summarize (trimming history)")
        recent = msgs[-RECENT_MESSAGES_KEEP:]
        old = msgs[:-RECENT_MESSAGES_KEEP]
        lines = []
        for m in old:
            role = getattr(m, "type", "user")
            content = getattr(m, "content", "")
            text = str(content)[:400]
            lines.append(f"{role}: {text}")
        history_text = "\n".join(lines)
        prompt = (
            "Summarize this coding session for an agent to continue.\n"
            "Focus on: task, repo, files touched, decisions. Only mention errors that appear explicitly in the history; do not invent or assume errors. Under 200 words.\n\n"
            f"History:\n{history_text}"
        )
        try:
            resp = invoke_with_retry(llm, prompt)
            summary = (getattr(resp, "content", "") or str(resp))[:1500]
        except Exception:
            summary = "(Summary unavailable)"
        summary_msg = HumanMessage(content=f"[Conversation summary]\n{summary}")
        return {"messages": Overwrite(value=[summary_msg] + list(recent))}

    def coder_node(state: AgentState) -> dict:
        logger.info("phase=coder")
        messages = state.get("messages") or []
        system = CODER_PROMPT
        if state.get("repo_path"):
            system += f"\n\nRepo: {state['repo_path']}"
        if state.get("files_changed"):
            system += f"\nFiles changed: {', '.join(state['files_changed'][:8])}"
        full = [SystemMessage(content=system)] + messages
        response = invoke_with_retry(coder_model, full)
        return {"messages": [response], "phase": "coding"}

    def run_tool_with_retry(name: str, state_dict: dict, args: dict) -> str:
        impl = tool_impls.get(name)
        if not impl:
            return "Error: Unknown tool; use a valid tool name."
        delay = TOOL_RETRY_INITIAL_DELAY
        last_exc = None
        for attempt in range(TOOL_MAX_RETRIES + 1):
            try:
                out = impl(state_dict, args)
                return out
            except TOOL_RETRY_EXCEPTIONS as e:
                last_exc = e
                if attempt >= TOOL_MAX_RETRIES:
                    return f"Error: {e} (after {TOOL_MAX_RETRIES + 1} attempts)"
                time.sleep(min(delay, 60))
                delay *= TOOL_RETRY_BACKOFF_FACTOR
            except Exception as e:
                return f"Error: {e}"
        return f"Error: {last_exc} (after {TOOL_MAX_RETRIES + 1} attempts)"

    def action_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        if not isinstance(last, AIMessage) or not getattr(last, "tool_calls", None):
            return {}
        tool_calls = last.tool_calls
        logger.info("phase=action tools=%s", [t.get("name") for t in tool_calls])
        results = []
        state_dict = dict(state)
        for tc in tool_calls:
            name = tc.get("name", "")
            args = tc.get("args") or {}
            tid = tc.get("id", "")
            content = run_tool_with_retry(name, state_dict, args)
            if isinstance(content, str) and content.startswith("Error:"):
                state_dict["last_tool_error"] = content[:500]
            if len(content) > MAX_TOOL_RESULT_CHARS:
                content = content[:MAX_TOOL_RESULT_CHARS] + "\n... (truncated)"
            results.append(ToolMessage(tool_call_id=tid, name=name, content=content))
        out: dict = {"messages": results}
        for k in ("repo_path", "repo_owner", "repo_name", "current_branch", "files_changed", "last_tool_error"):
            if k in state_dict and state_dict[k] is not None:
                out[k] = state_dict[k]
        return out

    def _run_cmd(cmd: str | list, cwd: str, timeout: int) -> tuple[bool, str]:
        if isinstance(cmd, str):
            cmd = ["sh", "-c", cmd]
        try:
            r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
            out = ((r.stdout or "") + "\n" + (r.stderr or "")).strip()
            if len(out) > TEST_OUTPUT_MAX_CHARS:
                out = out[-TEST_OUTPUT_MAX_CHARS:]
            return r.returncode == 0, out
        except subprocess.TimeoutExpired as e:
            return False, f"Timeout after {timeout}s: {e}"
        except Exception as e:
            return False, str(e)

    def tester_node(state: AgentState) -> dict:
        repo = state.get("repo_path") or ""
        if not repo:
            logger.info("phase=tester (no repo_path, skipping)")
            return {"tests_ok": False, "test_output": "No repo_path.", "phase": "coding"}
        messages = state.get("messages") or []
        files_changed = state.get("files_changed") or []
        context_parts = []
        for m in messages[-12:]:
            role = getattr(m, "type", "unknown")
            content = getattr(m, "content", "") or ""
            if isinstance(content, str) and len(content) > 600:
                content = content[:600] + "..."
            context_parts.append(f"{role}: {content}")
        context_parts.append(f"Files changed: {', '.join(files_changed) if files_changed else '(none)'}")
        context = "\n\n".join(context_parts)
        try:
            structured_llm = llm.with_structured_output(TestPlan)
            plan = invoke_with_retry(structured_llm, [
                SystemMessage(content=TESTER_SYSTEM),
                HumanMessage(content=f"Decide the test plan.\n\n{context}"),
            ])
        except Exception:
            plan = TestPlan(run_pytest=False, run_entry_point=bool(files_changed), entry_point_command="python main.py" if files_changed else None, reason="fallback")
        logger.info("phase=tester pytest=%s entry=%s cmd=%s", plan.run_pytest, plan.run_entry_point, plan.entry_point_command or "")
        if not plan.run_pytest and not plan.run_entry_point:
            return {"tests_ok": True, "test_output": "(No automated tests; critique-only.)", "phase": "git"}
        outputs = []
        all_ok = True
        if plan.run_pytest:
            ok, out = _run_cmd(["pytest", "-v", "--tb=short"], repo, TEST_PYTEST_TIMEOUT)
            outputs.append(f"[pytest]\n{out}")
            if not ok:
                all_ok = False
        if plan.run_entry_point and plan.entry_point_command:
            ok, out = _run_cmd(plan.entry_point_command, repo, TEST_ENTRY_POINT_TIMEOUT)
            outputs.append(f"[entry point: {plan.entry_point_command}]\n{out}")
            if not ok:
                all_ok = False
        test_output = "\n\n".join(outputs)
        if len(test_output) > TEST_OUTPUT_MAX_CHARS:
            test_output = test_output[-TEST_OUTPUT_MAX_CHARS:]
        if all_ok:
            return {"tests_ok": True, "test_output": test_output, "phase": "git"}
        fail_msg = HumanMessage(content=f"Tests/run failed:\n{test_output}\n\nFix the code and try again.")
        return {"tests_ok": False, "test_output": test_output, "messages": [fail_msg]}

    def git_ops_node(state: AgentState) -> dict:
        logger.info("phase=git_ops")
        messages = state.get("messages") or []
        system = GIT_PROMPT
        if state.get("files_changed"):
            system += f"\nFiles changed: {', '.join(state['files_changed'])}"
        system += "\nRespond with tool calls only (no plain-text summary)."
        full = [SystemMessage(content=system)] + messages
        response = invoke_with_retry(git_model, full)
        return {"messages": [response], "phase": "git"}

    def coder_has_tools(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage):
            tc = getattr(last, "tool_calls", None)
            if tc and len(tc) > 0:
                return "action"
        return "tester"

    def git_has_tools(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage):
            tc = getattr(last, "tool_calls", None)
            if tc and len(tc) > 0:
                return "action"
        return "end"

    def route_after_action(state: AgentState) -> str:
        return "git_ops" if state.get("phase") == "git" else "summarize"

    def route_after_tester(state: AgentState) -> str:
        return "git_ops" if state.get("tests_ok") else "summarize"

    graph = StateGraph(AgentState)
    graph.add_node("summarize", summarize_node)
    graph.add_node("coder", coder_node)
    graph.add_node("action", action_node)
    graph.add_node("tester", tester_node)
    graph.add_node("git_ops", git_ops_node)

    graph.set_entry_point("summarize")
    graph.add_edge("summarize", "coder")
    graph.add_conditional_edges("coder", coder_has_tools, {"action": "action", "tester": "tester"})
    graph.add_conditional_edges("action", route_after_action, {"summarize": "summarize", "git_ops": "git_ops"})
    graph.add_conditional_edges("tester", route_after_tester, {"summarize": "summarize", "git_ops": "git_ops"})
    graph.add_conditional_edges("git_ops", git_has_tools, {"action": "action", "end": END})

    return graph.compile(checkpointer=checkpointer or MemorySaver())
