from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage

from src.config import MESSAGE_WINDOW_BLOCKS


def trim_messages(messages: list[BaseMessage], system_content: str | None = None) -> list[BaseMessage]:
    out: list[BaseMessage] = []
    if system_content:
        out.append(SystemMessage(content=system_content))
    if not messages:
        return out
    blocks: list[list[BaseMessage]] = []
    current: list[BaseMessage] = []
    for m in messages:
        if isinstance(m, (AIMessage, HumanMessage)):
            if current and isinstance(current[-1], ToolMessage):
                blocks.append(current)
                current = []
            current.append(m)
        elif isinstance(m, ToolMessage):
            current.append(m)
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    kept = blocks[-MESSAGE_WINDOW_BLOCKS:] if len(blocks) > MESSAGE_WINDOW_BLOCKS else blocks
    for block in kept:
        out.extend(block)
    return out


def build_working_summary(state: dict) -> str:
    parts = []
    if state.get("working_summary"):
        parts.append(state["working_summary"])
    if state.get("repo_path"):
        parts.append(f"Repo: {state['repo_path']}")
    if state.get("current_branch"):
        parts.append(f"Branch: {state['current_branch']}")
    if state.get("files_changed"):
        parts.append(f"Files changed: {', '.join(state['files_changed'][:10])}")
    if state.get("last_tool_error"):
        parts.append(f"Last error: {state['last_tool_error']}")
    return " | ".join(parts) if parts else ""
