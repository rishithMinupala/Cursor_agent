from src.tools.codebase import (
    pull_repo_impl,
    get_appropriate_files_impl,
    get_appropriate_code_impl,
    read_code_impl,
    write_code_impl,
)
from src.tools.git import (
    create_branch_impl,
    commit_changes_impl,
    push_impl,
    create_pr_impl,
)
from src.tools.reflect import reflect_on_changes_impl, critique_changes_impl
from src.tools.testing import run_tests_impl

TOOL_IMPLS = {
    "pull_repo": pull_repo_impl,
    "get_appropriate_files": get_appropriate_files_impl,
    "get_appropriate_code": get_appropriate_code_impl,
    "read_code": read_code_impl,
    "write_code": write_code_impl,
    "create_branch": create_branch_impl,
    "commit_changes": commit_changes_impl,
    "push": push_impl,
    "create_pr": create_pr_impl,
    "reflect_on_changes": reflect_on_changes_impl,
    "critique_changes": critique_changes_impl,
    "run_tests": run_tests_impl,
}
