from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class PullRepoArgs(BaseModel):
    repo_url: str = Field(default="", description="Git clone URL. Omit to pull existing repo.")
    dest_path: str = Field(default="", description="Local path to clone into.")


class GrepSearchArgs(BaseModel):
    query: str = Field(description="Regex or plain-text pattern to search for in file contents.")
    case_sensitive: bool = Field(default=False, description="Whether the search is case sensitive.")
    include_pattern: str = Field(default="*.py", description="Comma-separated glob(s) of files to include (e.g. '*.py,*.ts').")
    exclude_pattern: str = Field(default="", description="Comma-separated glob(s) of files to exclude.")


class ReadCodeArgs(BaseModel):
    path: str = Field(description="Relative path to file from repo root.")


class WriteCodeArgs(BaseModel):
    path: str = Field(description="Relative path to file from repo root.")
    content: str = Field(description="Full new content for the file.")


class CreateDirArgs(BaseModel):
    path: str = Field(description="Relative path to directory from repo root. Parent dirs created as needed.")


class CreateFileArgs(BaseModel):
    path: str = Field(description="Relative path to new file from repo root. Fails if file already exists.")
    content: str = Field(description="Content to write.")


class EditInFileArgs(BaseModel):
    path: str = Field(description="Relative path to file from repo root.")
    edit_kind: str = Field(
        description="One of: replace (replace line range), insert_after (insert after at_line), insert_before (insert before at_line)."
    )
    content: str = Field(description="New content (for replace) or lines to insert.")
    start_line: int | None = Field(default=None, description="1-based start line for replace.")
    end_line: int | None = Field(default=None, description="1-based end line (inclusive) for replace.")
    at_line: int | None = Field(default=None, description="1-based line for insert_after/insert_before.")


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
        name="grep_search",
        description="Search inside repo files for a pattern (regex or plain text). Returns 'path:line:snippet' matches.",
        args_schema=GrepSearchArgs,
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
        name="create_dir",
        description="Create a directory (and parents). Path relative to repo root.",
        args_schema=CreateDirArgs,
    ),
    StructuredTool.from_function(
        func=_noop,
        name="create_file",
        description="Create a new file with content. Fails if file exists; use write_code to overwrite or edit_in_file to edit.",
        args_schema=CreateFileArgs,
    ),
    StructuredTool.from_function(
        func=_noop,
        name="edit_in_file",
        description="Edit existing file at a specific place: replace (start_line,end_line,content), insert_after (at_line,content), insert_before (at_line,content). Lines are 1-based.",
        args_schema=EditInFileArgs,
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

CODER_TOOL_NAMES = {
    "pull_repo", "grep_search", "read_code", "write_code",
    "create_dir", "create_file", "edit_in_file",
    "reflect_on_changes", "critique_changes",
}
GIT_TOOL_NAMES = {"create_branch", "commit_changes", "push", "create_pr"}

CODER_TOOLS = [t for t in TOOLS_FOR_LLM if t.name in CODER_TOOL_NAMES]
GIT_TOOLS = [t for t in TOOLS_FOR_LLM if t.name in GIT_TOOL_NAMES]
