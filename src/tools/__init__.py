from src.tools.codebase import (
    pull_repo_impl,
    read_code_impl,
    write_code_impl,
    create_dir_impl,
    create_file_impl,
    edit_in_file_impl,
)
from src.tools.git import (
    create_branch_impl,
    commit_changes_impl,
    push_impl,
    create_pr_impl,
)
from src.tools.reflect import reflect_on_changes_impl, critique_changes_impl
from src.tools.testing import run_tests_impl
from src.tools.search import grep_search_impl

TOOL_IMPLS = {
    "pull_repo": pull_repo_impl,
    "read_code": read_code_impl,
    "write_code": write_code_impl,
    "create_dir": create_dir_impl,
    "create_file": create_file_impl,
    "edit_in_file": edit_in_file_impl,
    "grep_search": grep_search_impl,
    "create_branch": create_branch_impl,
    "commit_changes": commit_changes_impl,
    "push": push_impl,
    "create_pr": create_pr_impl,
    "reflect_on_changes": reflect_on_changes_impl,
    "critique_changes": critique_changes_impl,
    "run_tests": run_tests_impl,
}
