import os
import re
from pathlib import Path
from typing import List

from src.config import IGNORE_DIRS, MAX_SNIPPET_CHARS, MAX_SNIPPETS


def grep_search_impl(state: dict, args: dict) -> str:
    """
    Search through repo files for a pattern (regex or plain text) and return bounded matches.

    Returns lines like:
    path/to/file.py:123: matched line content
    """
    repo = state.get("repo_path") or ""
    if not repo or not Path(repo).exists():
        return "Error: no repo_path. Call pull_repo first."

    query = args.get("query") or ""
    if not query:
        return "Error: query is required."

    case_sensitive = bool(args.get("case_sensitive", False))
    include_pattern = args.get("include_pattern") or ""
    exclude_pattern = args.get("exclude_pattern") or ""

    # Compile search pattern
    try:
        flags = 0 if case_sensitive else re.IGNORECASE
        pattern = re.compile(query, flags)
    except re.error as e:
        return f"Error: invalid regex pattern: {e}"

    # Convert simple glob-like patterns (*.py, *.js, etc.) to regex
    def _glob_to_regex(glob_str: str) -> List[re.Pattern]:
        regexes: List[re.Pattern] = []
        for raw in glob_str.split(","):
            g = raw.strip()
            if not g:
                continue
            # minimal glob support: *, ?, .
            r = (
                g.replace(".", r"\.")
                .replace("*", r".*")
                .replace("?", r".")
            )
            try:
                regexes.append(re.compile(f"^{r}$"))
            except re.error:
                continue
        return regexes

    include_regexes = _glob_to_regex(include_pattern) if include_pattern else None
    exclude_regexes = _glob_to_regex(exclude_pattern) if exclude_pattern else None

    root = Path(repo)
    results: list[str] = []

    for dirpath, dirnames, filenames in os.walk(root):
        # skip ignored dirs
        rel_parts = Path(dirpath).relative_to(root).parts
        if any(part in IGNORE_DIRS for part in rel_parts):
            continue

        for filename in filenames:
            if filename in IGNORE_DIRS:
                continue
            if include_regexes and not any(r.match(filename) for r in include_regexes):
                continue
            if exclude_regexes and any(r.match(filename) for r in exclude_regexes):
                continue

            fp = Path(dirpath) / filename
            rel = fp.relative_to(root)
            try:
                with fp.open("r", encoding="utf-8", errors="ignore") as f:
                    for i, line in enumerate(f, 1):
                        if pattern.search(line):
                            snippet = line.rstrip()
                            if len(snippet) > MAX_SNIPPET_CHARS:
                                snippet = snippet[:MAX_SNIPPET_CHARS] + "..."
                            results.append(f"{rel}:{i}: {snippet}")
                            if len(results) >= MAX_SNIPPETS * 5:
                                break
                if len(results) >= MAX_SNIPPETS * 5:
                    break
            except Exception:
                continue
        if len(results) >= MAX_SNIPPETS * 5:
            break

    if not results:
        return "No matches found."
    return "\n".join(results)

