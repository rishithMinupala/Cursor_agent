from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class PullRepoArgs(BaseModel):
    repo_url: str = Field(default="", description="Git clone URL. Omit to pull existing repo.")
    dest_path: str = Field(default="", description="Local path to clone into.")


class GetAppropriateFilesArgs(BaseModel):
    query: str = Field(description="Search query to find relevant files (e.g. feature name, module).")


class GetAppropriateCodeArgs(BaseModel):
    query: str = Field(description="Search query to find relevant code snippets.")


class ReadCodeArgs(BaseModel):
    path: str = Field(description="Relative path to file from repo root.")


class WriteCodeArgs(BaseModel):
    path: str = Field(description="Relative path to file from repo root.")
    content: str = Field(description="Full new content for the file.")


class CreateBranchArgs(BaseModel):
    branch_name: str = Field(description="Branch name, e.g. 'add-login'.")
    type: str = Field(default="feature", description="'feature' or 'bugfix'.")
    base_branch: str = Field(default="main", description="Base branch to create from (e.g. main, dev).")


class CommitChangesArgs(BaseModel):
    message: str = Field(description="Commit message.")


class PushArgs(BaseModel):
    pass


class CreatePRArgs(BaseModel):
    title: str = Field(description="PR title.")
    body: str = Field(default="", description="PR description/body.")
    base_branch: str = Field(default="main", description="Target branch to merge into (e.g. main, dev).")


class ReflectOnChangesArgs(BaseModel):
    files_changed: list[str] = Field(default_factory=list, description="Paths of files changed.")
    summary: str = Field(default="", description="Short summary of changes.")


class CritiqueChangesArgs(BaseModel):
    files_changed: list[str] = Field(default_factory=list, description="Paths of files changed.")
    summary: str = Field(default="", description="Short summary of changes.")


class RunTestsArgs(BaseModel):
    command: str = Field(default="pytest", description="Test command, e.g. 'pytest' or 'npm test'.")


def _noop(**kwargs) -> str:
    return ""


TOOLS_FOR_LLM = [
    StructuredTool.from_function(
        func=_noop,
        name="pull_repo",
        description="Clone a repo (first time) or pull latest. Builds codebase index. Required before other code tools.",
        args_schema=PullRepoArgs,
    ),
    StructuredTool.from_function(
        func=_noop,
        name="get_appropriate_files",
        description="Find files relevant to a query. Returns file paths and one-line preview. Use before read_code to avoid loading too much.",
        args_schema=GetAppropriateFilesArgs,
    ),
    StructuredTool.from_function(
        func=_noop,
        name="get_appropriate_code",
        description="Search for code snippets matching a query. Returns bounded snippets.",
        args_schema=GetAppropriateCodeArgs,
    ),
    StructuredTool.from_function(
        func=_noop,
        name="read_code",
        description="Read contents of a single file. Path relative to repo root.",
        args_schema=ReadCodeArgs,
    ),
    StructuredTool.from_function(
        func=_noop,
        name="write_code",
        description="Write or overwrite a file. Path relative to repo root.",
        args_schema=WriteCodeArgs,
    ),
    StructuredTool.from_function(
        func=_noop,
        name="create_branch",
        description="Create and checkout a new branch (feature or bugfix).",
        args_schema=CreateBranchArgs,
    ),
    StructuredTool.from_function(
        func=_noop,
        name="commit_changes",
        description="Stage all changes and commit with the given message.",
        args_schema=CommitChangesArgs,
    ),
    StructuredTool.from_function(
        func=_noop,
        name="push",
        description="Push current branch to origin.",
        args_schema=PushArgs,
    ),
    StructuredTool.from_function(
        func=_noop,
        name="create_pr",
        description="Open a pull request. Call after push. Requires gh CLI.",
        args_schema=CreatePRArgs,
    ),
    StructuredTool.from_function(
        func=_noop,
        name="reflect_on_changes",
        description="Reflect on your edits: what was done, does it match the task, any risks. Call before commit.",
        args_schema=ReflectOnChangesArgs,
    ),
    StructuredTool.from_function(
        func=_noop,
        name="critique_changes",
        description="Get a critique of your code changes. Call before commit.",
        args_schema=CritiqueChangesArgs,
    ),
    StructuredTool.from_function(
        func=_noop,
        name="run_tests",
        description="Run tests (e.g. pytest). Call before commit to validate.",
        args_schema=RunTestsArgs,
    ),
]
