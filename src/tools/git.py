import os
import subprocess
from pathlib import Path

from src.services.github_api import GitHubApiClient


def _run(cmd: list[str], cwd: str, capture=True) -> tuple[bool, str]:
    try:
        r = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=capture,
            text=True,
            timeout=60,
        )
        out = (r.stdout or "").strip() or (r.stderr or "").strip()
        return r.returncode == 0, out or str(r.returncode)
    except Exception as e:
        return False, str(e)


def create_branch_impl(state: dict, args: dict) -> str:
    repo = state.get("repo_path") or ""
    if not repo or not Path(repo).exists():
        return "Error: no repo_path. Call pull_repo first."
    name = args.get("branch_name") or args.get("name") or ""
    branch_type = (args.get("type") or "feature").lower()
    base_branch = args.get("base_branch") or "main"
    if not name:
        return "Error: branch_name required."
    prefix = "feature" if branch_type == "feature" else "bugfix"
    full_name = f"{prefix}/{name}" if "/" not in name else name

    client = GitHubApiClient()
    if client.token and state.get("repo_owner") and state.get("repo_name"):
        base_sha = client.get_branch_sha(state["repo_owner"], state["repo_name"], base_branch)
        if not base_sha:
            return f"Error: could not get SHA for base branch {base_branch}."
        ok = client.create_branch(state["repo_owner"], state["repo_name"], full_name, base_sha)
        if ok:
            state["current_branch"] = full_name
            return f"Created branch {full_name} (API)."
        return "Error: failed to create branch via API."
    ok, out = _run(["git", "checkout", "-b", full_name], repo)
    if ok:
        state["current_branch"] = full_name
        return f"Created and checked out branch: {full_name}"
    return f"Error: {out}"


def commit_changes_impl(state: dict, args: dict) -> str:
    repo = state.get("repo_path") or ""
    if not repo or not Path(repo).exists():
        return "Error: no repo_path."
    message = args.get("message") or args.get("commit_message") or "Update"
    files_changed = state.get("files_changed") or []

    client = GitHubApiClient()
    if client.token and state.get("repo_owner") and state.get("repo_name") and state.get("current_branch"):
        if not files_changed:
            return "No files changed to upload."
        uploaded = []
        for rel_path in files_changed:
            fp = Path(repo) / rel_path
            if not fp.is_file():
                continue
            content = fp.read_bytes()
            path_str = str(rel_path).replace("\\", "/")
            if client.upload_file(
                state["repo_owner"],
                state["repo_name"],
                state["current_branch"],
                path_str,
                content,
                message,
            ):
                uploaded.append(path_str)
        if uploaded:
            return f"Uploaded {len(uploaded)} file(s) via API: {', '.join(uploaded)}"
        return "Error: no files uploaded via API."
    ok, out = _run(["git", "add", "-A"], repo)
    if not ok:
        return f"Error git add: {out}"
    ok, out = _run(["git", "commit", "-m", message], repo)
    if not ok:
        if "nothing to commit" in out.lower():
            return "Nothing to commit (working tree clean)."
        return f"Error: {out}"
    return f"Committed: {message}"


def push_impl(state: dict, args: dict) -> str:
    if GitHubApiClient().token and state.get("repo_owner") and state.get("repo_name"):
        return "Using GitHub API; push not needed."
    repo = state.get("repo_path") or ""
    if not repo or not Path(repo).exists():
        return "Error: no repo_path."
    branch = state.get("current_branch") or "HEAD"
    ok, out = _run(["git", "push", "-u", "origin", branch], repo)
    if ok:
        return f"Pushed branch {branch}."
    return f"Error: {out}"


def create_pr_impl(state: dict, args: dict) -> str:
    title = args.get("title") or ""
    body = args.get("body") or args.get("description") or ""
    base_branch = args.get("base_branch") or "main"
    if not title:
        return "Error: title required for PR."
    repo = state.get("repo_path") or ""
    if not repo:
        return "Error: no repo_path."

    client = GitHubApiClient()
    if client.token and state.get("repo_owner") and state.get("repo_name") and state.get("current_branch"):
        url = client.create_pr(
            state["repo_owner"],
            state["repo_name"],
            state["current_branch"],
            base_branch,
            title,
            body,
        )
        if url:
            return f"PR created: {url}"
        return "Error: failed to create PR via API."
    cmd = ["gh", "pr", "create", "--title", title]
    if body:
        cmd.extend(["--body", body])
    ok, out = _run(cmd, repo)
    if ok:
        return f"PR created: {out}"
    return f"Error: {out}"
