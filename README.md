# PR Agent

Agentic workflow: natural language task → code changes → validation → PR.

## Setup

```bash
pip install -r requirements.txt
# .env: GEMINI_API_KEY (preferred), or OPENAI_API_KEY / ANTHROPIC_API_KEY
# Optional: GITHUB_TOKEN for GitHub API (branch, upload files, create PR without gh CLI)
```

## Run

Put the repo URL in the task; the agent will call `pull_repo` when needed.

```bash
python run_agent.py --task "Clone https://github.com/owner/repo.git, add a hello() in main.py, then open a PR"
```

Model: Gemini (default if `GEMINI_API_KEY` set), else Anthropic, else OpenAI. Override with `--model gemini-2.0-flash`.

## Architecture

- **State**: `messages`, `working_summary`, `repo_path`, `repo_owner`, `repo_name`, `codebase_index`, `current_branch`, `files_changed`, `last_tool_error`
- **Graph**: single agent node (LLM + tools) → action node (run tool impls with state) → loop until no tool calls
- **Context**: messages trimmed to last N blocks; working summary injected into system prompt
- **Tools**: pull_repo, get_appropriate_files, get_appropriate_code, read_code, write_code, create_branch, commit_changes, push, create_pr, reflect_on_changes, critique_changes, run_tests

**Git**: With `GITHUB_TOKEN`, branch/upload/PR use GitHub REST API (no `gh` CLI). Without token, git/gh CLI is used.
