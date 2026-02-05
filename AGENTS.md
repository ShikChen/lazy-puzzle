# AGENTS.md

Guidelines for AI coding agents working in this repository:

- Run Python commands via `uv run`.
- Add dependencies with `uv add`.
- Format before committing: `uv run ruff format`.
- Check with `uv run ruff check` and `uv run ty check`.
- Use Conventional Commits for commit messages (e.g., `feat: ...`, `fix: ...`).
- Prefer clear dataclasses over opaque tuples when returning structured state.
- Keep changes focused and update docs/tests when behavior changes.
