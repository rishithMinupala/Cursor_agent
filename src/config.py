from pathlib import Path

DEFAULT_REPO_DIR = Path.cwd() / "repo"
MAX_FILE_CHARS = 4000
MAX_SNIPPETS = 10
MAX_SNIPPET_CHARS = 500
MAX_FILES_IN_LIST = 25
MAX_TOOL_RESULT_CHARS = 8000
MESSAGE_WINDOW_BLOCKS = 8
REFLECT_MAX_OUTPUT_CHARS = 1000
TEST_OUTPUT_TAIL_CHARS = 2000
IGNORE_DIRS = {".git", "__pycache__", "node_modules", "venv", ".venv", "dist", "build", ".env"}
