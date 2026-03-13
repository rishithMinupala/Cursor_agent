from typing import TypedDict, Annotated, Any
import operator
from langchain_core.messages import AnyMessage


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], operator.add]
    working_summary: str
    repo_path: str
    repo_owner: str
    repo_name: str
    codebase_index: list[dict[str, Any]]
    current_branch: str
    files_changed: list[str]
    last_tool_error: str
