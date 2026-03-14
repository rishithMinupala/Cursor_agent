import subprocess
from pathlib import Path

from src.services.github_api import parse_repo_url
from src.config import DEFAULT_REPO_DIR, MAX_FILE_CHARS


def pull_repo_impl(state: dict, args: dict) -> str:
    url = args.get("repo_url") or ""
    dest = args.get("dest_path") or str(DEFAULT_REPO_DIR)
    if not url:
        if state.get("repo_path") and Path(state["repo_path"]).exists():
            try:
                subprocess.run(
                    ["git", "pull"],
                    cwd=state["repo_path"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                return "Pulled latest."
            except subprocess.CalledProcessError as e:
                return f"Error: {e.stderr or str(e)}"
        return "Error: repo_url required for first clone, or set repo_path in state."
    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(dest_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        state["repo_path"] = str(dest_path)
        owner, name = parse_repo_url(url)
        if owner and name:
            state["repo_owner"] = owner
            state["repo_name"] = name
        return f"Cloned to {dest_path}."
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr or str(e)}"


def read_code_impl(state: dict, args: dict) -> str:
    repo = state.get("repo_path") or ""
    path_arg = args.get("path") or args.get("file_path") or ""
    if not repo or not Path(repo).exists():
        return "Error: no repo_path. Call pull_repo first."
    if not path_arg:
        return "Error: path or file_path required."
    fp = Path(repo) / path_arg
    if not fp.is_file():
        return f"Error: not a file or not found: {path_arg}"
    try:
        content = fp.read_text(errors="ignore")
        if len(content) > MAX_FILE_CHARS:
            content = content[:MAX_FILE_CHARS] + "\n... truncated"
        return content
    except Exception as e:
        return f"Error: {e}"


def write_code_impl(state: dict, args: dict) -> str:
    repo = state.get("repo_path") or ""
    path_arg = args.get("path") or args.get("file_path") or ""
    content = args.get("content") or ""
    if not repo or not Path(repo).exists():
        return "Error: no repo_path. Call pull_repo first."
    if not path_arg:
        return "Error: path or file_path required."
    fp = Path(repo) / path_arg
    fp.parent.mkdir(parents=True, exist_ok=True)
    try:
        fp.write_text(content, encoding="utf-8")
        _track_changed(state, path_arg)
        return f"Updated {path_arg}"
    except Exception as e:
        return f"Error: {e}"


def _track_changed(state: dict, path_arg: str) -> None:
    changed = state.get("files_changed") or []
    if path_arg not in changed:
        changed.append(path_arg)
    state["files_changed"] = changed


def create_dir_impl(state: dict, args: dict) -> str:
    repo = state.get("repo_path") or ""
    path_arg = (args.get("path") or args.get("dir_path") or "").strip().rstrip("/")
    if not repo or not Path(repo).exists():
        return "Error: no repo_path. Call pull_repo first."
    if not path_arg:
        return "Error: path or dir_path required."
    fp = Path(repo) / path_arg
    try:
        fp.mkdir(parents=True, exist_ok=True)
        return f"Created directory {path_arg}"
    except Exception as e:
        return f"Error: {e}"


def create_file_impl(state: dict, args: dict) -> str:
    repo = state.get("repo_path") or ""
    path_arg = args.get("path") or args.get("file_path") or ""
    content = args.get("content") or ""
    if not repo or not Path(repo).exists():
        return "Error: no repo_path. Call pull_repo first."
    if not path_arg:
        return "Error: path or file_path required."
    fp = Path(repo) / path_arg
    if fp.exists() and fp.is_file():
        return f"Error: file already exists: {path_arg}. Use write_code to overwrite or edit_in_file to edit."
    fp.parent.mkdir(parents=True, exist_ok=True)
    try:
        fp.write_text(content, encoding="utf-8")
        _track_changed(state, path_arg)
        return f"Created file {path_arg}"
    except Exception as e:
        return f"Error: {e}"


def edit_in_file_impl(state: dict, args: dict) -> str:
    repo = state.get("repo_path") or ""
    path_arg = args.get("path") or args.get("file_path") or ""
    kind = (args.get("edit_kind") or args.get("kind") or "replace").strip().lower()
    content = args.get("content") or ""
    start_line = args.get("start_line")
    end_line = args.get("end_line")
    at_line = args.get("at_line")
    if not repo or not Path(repo).exists():
        return "Error: no repo_path. Call pull_repo first."
    if not path_arg:
        return "Error: path or file_path required."
    fp = Path(repo) / path_arg
    if not fp.is_file():
        return f"Error: not a file or not found: {path_arg}"
    try:
        raw = fp.read_text(encoding="utf-8")
        lines = raw.splitlines(keepends=True)
        if not lines:
            lines = [""]
    except Exception as e:
        return f"Error reading file: {e}"
    n = len(lines)
    new_lines = content.splitlines()
    if kind == "replace":
        if start_line is None or end_line is None:
            return "Error: start_line and end_line required for edit_kind=replace."
        s, e = int(start_line), int(end_line)
        if s < 1 or e < 1 or s > e:
            return "Error: start_line and end_line must be 1-based with start_line <= end_line."
        if e > n:
            return f"Error: end_line {e} exceeds file length {n}."
        head = lines[: s - 1]
        tail = lines[e:]
        mid = [line + "\n" if not line.endswith("\n") else line for line in new_lines] if new_lines else []
        lines = head + mid + tail
    elif kind in ("insert_after", "insert_before"):
        if at_line is None:
            return "Error: at_line required for insert_after/insert_before."
        pos = int(at_line)
        if pos < 1:
            return "Error: at_line must be >= 1."
        insert = [line + "\n" if not line.endswith("\n") else line for line in new_lines] if new_lines else []
        if kind == "insert_after":
            if pos > n:
                lines = lines + insert
            else:
                lines = lines[:pos] + insert + lines[pos:]
        else:
            lines = lines[: pos - 1] + insert + lines[pos - 1 :]
    else:
        return "Error: edit_kind must be replace, insert_after, or insert_before."
    try:
        fp.write_text("".join(lines), encoding="utf-8")
        _track_changed(state, path_arg)
        return f"Edited {path_arg} ({kind})"
    except Exception as e:
        return f"Error: {e}"
