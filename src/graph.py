from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import ToolMessage, AIMessage

from src.state import AgentState
from src.context import trim_messages, build_working_summary
from src.tools import TOOL_IMPLS
from src.tools.reflect import reflect_on_changes_impl, critique_changes_impl
from src.tools.schemas import TOOLS_FOR_LLM
from src.config import MAX_TOOL_RESULT_CHARS

SYSTEM_PROMPT = """You are a coding agent that implements features or bugfixes and opens pull requests.

Workflow:
1. Clone or pull the repo (pull_repo) so you have the codebase.
2. Understand the task: use get_appropriate_files and get_appropriate_code to find relevant code, then read_code for full file contents.
3. Make changes with write_code. Prefer targeted edits; you can read a file, then write_code with the full updated content.
4. Optionally run_tests to validate. If tests fail, fix and re-run.
5. Reflect on your changes (reflect_on_changes) and get a critique (critique_changes). If issues are found, fix them.
6. When satisfied: create_branch (feature or bugfix), commit_changes, push, then create_pr with a clear title and description.

Use the working context below to know current repo path, branch, and files changed. If a tool returns an error, retry or report. Prefer get_appropriate_files before reading many files to avoid loading too much. After creating the PR, respond to the user that the PR is ready."""


def create_graph(llm, checkpointer=None):
    model_with_tools = llm.bind_tools(TOOLS_FOR_LLM)
    tool_impls = dict(TOOL_IMPLS)
    tool_impls["reflect_on_changes"] = lambda s, a: reflect_on_changes_impl(s, a, llm)
    tool_impls["critique_changes"] = lambda s, a: critique_changes_impl(s, a, llm)

    def agent_node(state: AgentState) -> dict:
        working = build_working_summary(state)
        system = f"{SYSTEM_PROMPT}\n\nCurrent context: {working}" if working else SYSTEM_PROMPT
        messages = trim_messages(state["messages"], system_content=system)
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    def action_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        if not isinstance(last, AIMessage) or not getattr(last, "tool_calls", None):
            return {"messages": []}
        tool_calls = last.tool_calls
        results = []
        state_dict = dict(state)
        for tc in tool_calls:
            name = tc.get("name", "")
            args = tc.get("args") or {}
            tid = tc.get("id", "")
            if name not in tool_impls:
                content = "Unknown tool; retry with a valid tool name."
            else:
                try:
                    content = tool_impls[name](state_dict, args)
                except Exception as e:
                    content = f"Error: {e}"
            if isinstance(content, str) and content.startswith("Error:"):
                state_dict["last_tool_error"] = content[:500]
            if len(content) > MAX_TOOL_RESULT_CHARS:
                content = content[:MAX_TOOL_RESULT_CHARS] + "\n... (truncated)"
            results.append(ToolMessage(tool_call_id=tid, name=name, content=content))
        out: dict = {"messages": results}
        for k in ("repo_path", "repo_owner", "repo_name", "codebase_index", "current_branch", "files_changed", "last_tool_error"):
            if k in state_dict and state_dict[k] is not None:
                out[k] = state_dict[k]
        return out

    def has_tool_calls(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "action"
        return "end"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("action", action_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", has_tool_calls, {"action": "action", "end": END})
    graph.add_edge("action", "agent")
    return graph.compile(checkpointer=checkpointer or MemorySaver())
