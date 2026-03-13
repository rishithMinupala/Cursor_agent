import os
import base64
import re
import requests
from pathlib import Path
from typing import Optional


def _parse_repo_url(url: str) -> tuple[Optional[str], Optional[str]]:
    if not url:
        return None, None
    m = re.match(r"(?:https?://|git@)github\.com[:/]([^/]+)/([^/.]+)(?:\.git)?", url.strip())
    if m:
        return m.group(1), m.group(2)
    return None, None


class GitHubApiClient:
    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv("GITHUB_TOKEN")
        self.session = requests.Session()
        if self.token:
            self.session.headers.update({
                "Authorization": f"token {self.token}",
                "Accept": "application/vnd.github.v3+json",
            })

    def _base_url(self, owner: str, repo: str) -> str:
        return f"https://api.github.com/repos/{owner}/{repo}"

    def get_branch_sha(self, owner: str, repo: str, branch: str) -> Optional[str]:
        url = f"{self._base_url(owner, repo)}/git/ref/heads/{branch}"
        try:
            r = self.session.get(url, timeout=30)
            r.raise_for_status()
            return r.json()["object"]["sha"]
        except Exception:
            return None

    def create_branch(self, owner: str, repo: str, branch_name: str, base_sha: str) -> bool:
        url = f"{self._base_url(owner, repo)}/git/refs"
        payload = {"ref": f"refs/heads/{branch_name}", "sha": base_sha}
        try:
            r = self.session.post(url, json=payload, timeout=30)
            if r.status_code in (200, 201):
                return True
            if r.status_code == 422 and "already exists" in (r.text or "").lower():
                return True
            return False
        except Exception:
            return False

    def upload_file(
        self,
        owner: str,
        repo: str,
        branch: str,
        file_path: str,
        content: bytes,
        message: str = "Update file",
    ) -> bool:
        url = f"{self._base_url(owner, repo)}/contents/{file_path}"
        payload = {
            "message": message,
            "content": base64.b64encode(content).decode(),
            "branch": branch,
        }
        try:
            r = self.session.put(url, json=payload, timeout=30)
            return r.status_code in (200, 201)
        except Exception:
            return False

    def create_pr(
        self,
        owner: str,
        repo: str,
        head_branch: str,
        base_branch: str,
        title: str,
        body: str = "",
    ) -> Optional[str]:
        url = f"{self._base_url(owner, repo)}/pulls"
        payload = {
            "title": title,
            "head": head_branch,
            "base": base_branch,
            "body": body,
        }
        try:
            r = self.session.post(url, json=payload, timeout=30)
            r.raise_for_status()
            return r.json().get("html_url")
        except Exception:
            return None


def parse_repo_url(url: str) -> tuple[Optional[str], Optional[str]]:
    return _parse_repo_url(url)
