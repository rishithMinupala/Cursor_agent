import os
import subprocess
from pathlib import Path

from src.services.github_api import parse_repo_url
from src.config import (
    DEFAULT_REPO_DIR,
    MAX_FILE_CHARS,
    MAX_SNIPPETS,
    MAX_SNIPPET_CHARS,
    MAX_FILES_IN_LIST,
    IGNORE_DIRS,
)


def _build_index(repo_path: str) -> list[dict]:
    root = Path(repo_path)
    if not root.exists():
        return []
    index = []
    for p in root.rglob("*"):
        if p.is_file() and p.name not in IGNORE_DIRS and p.suffix:
            rel = p.relative_to(root)
            if any(part in IGNORE_DIRS for part in rel.parts):
                continue
            try:
                line = p.read_text(errors="ignore").split("\n")[0][:80]
            except Exception:
                line = ""
            index.append({"path": str(rel), "first_line": line})
    return index


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
                state["codebase_index"] = _build_index(state["repo_path"])
                return "Pulled latest. Index updated."
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
        state["codebase_index"] = _build_index(str(dest_path))
        owner, name = parse_repo_url(url)
        if owner and name:
            state["repo_owner"] = owner
            state["repo_name"] = name
        return f"Cloned to {dest_path}. Index built."
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr or str(e)}"


def get_appropriate_files_impl(state: dict, args: dict) -> str:
    repo = state.get("repo_path") or ""
    if not repo or not Path(repo).exists():
        return "Error: no repo_path or repo not found. Call pull_repo first."
    query = (args.get("query") or "").lower()
    index = state.get("codebase_index") or _build_index(repo)
    state["codebase_index"] = index
    matches = []
    for item in index:
        path = item["path"].lower()
        first = (item.get("first_line") or "").lower()
        if query in path or query in first:
            matches.append(f"{item['path']}: {item.get('first_line', '')[:60]}")
    out = matches[:MAX_FILES_IN_LIST]
    return "\n".join(out) if out else "No matching files."


def get_appropriate_code_impl(state: dict, args: dict) -> str:
    repo = state.get("repo_path") or ""
    if not repo or not Path(repo).exists():
        return "Error: no repo_path. Call pull_repo first."
    query = (args.get("query") or "").lower()
    index = state.get("codebase_index") or _build_index(repo)
    state["codebase_index"] = index
    snippets = []
    root = Path(repo)
    for item in index:
        if len(snippets) >= MAX_SNIPPETS:
            break
        path = item["path"].lower()
        if query not in path and query not in (item.get("first_line") or "").lower():
            continue
        fp = root / item["path"]
        if not fp.is_file():
            continue
        try:
            content = fp.read_text(errors="ignore")
            chunk = content[:MAX_SNIPPET_CHARS]
            if len(content) > MAX_SNIPPET_CHARS:
                chunk += "\n... truncated"
            snippets.append(f"--- {item['path']} ---\n{chunk}")
        except Exception as e:
            snippets.append(f"--- {item['path']} ---\nError: {e}")
    return "\n\n".join(snippets) if snippets else "No matching code."


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
        changed = state.get("files_changed") or []
        if path_arg not in changed:
            changed.append(path_arg)
        state["files_changed"] = changed
        return f"Updated {path_arg}"
    except Exception as e:
        return f"Error: {e}"
