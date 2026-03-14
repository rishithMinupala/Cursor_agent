from typing import TypedDict, Annotated, Any
import operator
from langchain_core.messages import AnyMessage


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], operator.add]
    repo_path: str
    repo_owner: str
    repo_name: str
    current_branch: str
    files_changed: list[str]
    last_tool_error: str
    phase: str
    tests_ok: bool
    test_output: str
