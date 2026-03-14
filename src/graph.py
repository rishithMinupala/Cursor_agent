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
from src.tools.reflect import critique_changes_impl, reflect_on_changes_impl
from src.tools.schemas import CODER_TOOLS, GIT_TOOLS

logger = logging.getLogger(__name__)


class TestPlan(BaseModel):
    run_pytest: bool = Field(description="True if pytest should be run (repo has tests and code was changed).")
    run_entry_point: bool = Field(description="True if the app should be run from its entry point to check for startup/runtime errors.")
    entry_point_command: str | None = Field(default=None, description="Shell command to run the app, e.g. 'python -m src.main' or 'python main.py'. Required when run_entry_point is True.")
    reason: str | None = Field(default=None, description="Brief reason for this plan.")

CODER_PROMPT = """You are a coding agent. Implement the user's feature or bugfix.

Workflow:
1. Use pull_repo to clone if needed.
2. Use grep_search to find relevant code, read_code to read files.
3. For new files: create_dir if needed, then create_file(path, content).
4. After creating a new file you must integrate it: identify which existing files should use it (e.g. main entrypoint, __init__.py, callers), read_code those files, then use edit_in_file (or write_code) to add imports and wire the new code so the feature actually runs. Do not stop after create_file; integration is required.
5. For other edits in existing files: use edit_in_file with edit_kind replace (start_line, end_line, content) or insert_after/insert_before (at_line, content). Lines are 1-based. Use write_code only for full-file overwrite.
6. Use reflect_on_changes and critique_changes before finishing.
7. When done, respond briefly (e.g. "Done with code changes."). Do not call git tools - the workflow will run tests then git for you."""

GIT_PROMPT = """You are a git/PR agent. Create branch, commit, push, and open a PR for the changes.

Use create_branch, commit_changes, push, create_pr. Generate a clear PR title and body from the task context."""

TESTER_SYSTEM = """You are a testing agent. Given the conversation (user task, what the coder did) and the list of changed files, decide how to verify the changes.

Rules:
- If ONLY docs/config were changed (e.g. only .md, .txt, .yml, .yaml, .json, .toml, .env): set run_pytest=false and run_entry_point=false. Critique was already done; no automated test needed.
- If application code was changed (.py, .js, .ts, etc.): set run_entry_point=true and provide the shell command to run the app from its entry point (e.g. "python -m src.main", "python main.py", "node index.js"). The coder may have indicated the entry point in the conversation; otherwise infer from common patterns (main.py, __main__.py, package.json main, etc.).
- Set run_pytest=true only if the repo likely has tests and the changes are testable code (e.g. Python under src/ or app/). If there are no tests or only docs changed, set run_pytest=false.
- entry_point_command must be a single command runnable from the repo root (e.g. "python -m src.main" or "python main.py"). No shell operators like && or |."""


def create_graph(llm, checkpointer=None):
    coder_model = llm.bind_tools(CODER_TOOLS)
    git_model = llm.bind_tools(GIT_TOOLS)
    tool_impls = dict(TOOL_IMPLS)
    tool_impls["reflect_on_changes"] = lambda s, a: reflect_on_changes_impl(s, a, llm)
    tool_impls["critique_changes"] = lambda s, a: critique_changes_impl(s, a, llm)

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
            "Focus on: task, repo, files touched, decisions, errors. Under 200 words.\n\n"
            f"History:\n{history_text}"
        )
        try:
            resp = llm.invoke(prompt)
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
        response = coder_model.invoke(full)
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
            plan = structured_llm.invoke([
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
        full = [SystemMessage(content=system)] + messages
        response = git_model.invoke(full)
        return {"messages": [response], "phase": "git"}

    def coder_has_tools(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "action"
        return "tester"

    def git_has_tools(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
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
