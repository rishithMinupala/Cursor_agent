import subprocess
from pathlib import Path

from src.config import TEST_OUTPUT_TAIL_CHARS


def run_tests_impl(state: dict, args: dict) -> str:
    repo = state.get("repo_path") or ""
    if not repo or not Path(repo).exists():
        return "Error: no repo_path. Call pull_repo first."
    command = args.get("command") or "pytest"
    if isinstance(command, str):
        cmd = command.split()
    else:
        cmd = list(command)
    if not cmd:
        cmd = ["pytest"]
    try:
        r = subprocess.run(
            cmd,
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=120,
        )
        out = (r.stdout or "") + "\n" + (r.stderr or "")
        if len(out) > TEST_OUTPUT_TAIL_CHARS:
            out = "... (truncated)\n" + out[-TEST_OUTPUT_TAIL_CHARS:]
        status = "PASSED" if r.returncode == 0 else "FAILED"
        return f"Tests {status} (exit code {r.returncode}):\n{out}"
    except subprocess.TimeoutExpired:
        return "Error: tests timed out."
    except Exception as e:
        return f"Error: {e}"
