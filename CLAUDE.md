## Workflow Orchestration

### 1. Plan Node Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately – don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes – don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests – then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

## Task Management

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.
- **DRY**: Don't repeat yourself. Architect code for reuse as much as is practical.

## Python-Specific Instructions

- **PEP8**: Ensure that all code is PEP8 compliant.
- **Object-Orientation**: Use object-oriented principles when designing applications and writing code.
- **Virtual Environment**: Always use a virtual environment, installed in a folder called `.venv`.
- **Code Formatting**: Format code using the Black code formatter.
- **Linting**: Use `ruff` for linting alongside Black.
- **Type Safety**: Annotate all functions and methods with type hints (PEP 484); use `mypy` or `pyright` for static type checking.
- **FastAPI**: Always use FastAPI for REST API development.
- **FastMCP**: Always use FastMCP for MCP implementation.
- **REST First**: If the project specifies an MCP implementation, always build FastAPI REST endpoints first, then wrap those endpoints with FastMCP.
- **Async**: Use `async`/`await` patterns throughout FastAPI route handlers and service layers.
- **Pydantic Models**: Define all data schemas as Pydantic models; avoid passing raw `dict` between layers.
- **Error Handling**: Define custom exception classes for domain-specific errors rather than raising generic exceptions.
- **Logging**: Use Python's built-in `logging` module instead of `print` statements; configure log levels via environment variables.
- **Environment Variables**: Use `.env` files for environment variables during development; never commit them — always include in `.gitignore` and provide a `.env.example` template.
- **Security**: Validate and sanitize all external inputs, especially in FastAPI request models using Pydantic validators.
- **Requirements**: Use a `requirements.txt` file for production dependencies and a `requirements-dev.txt` for development-only dependencies (e.g., pytest, Black, ruff). Pin exact versions via `pip freeze`. Keep both files up to date at all times.
- **Testing**: Use `pytest` for all unit and integration tests; organize tests in a `tests/` directory mirroring the source structure; enforce a minimum 80% coverage threshold via `pytest-cov`.
- **Project Structure**: Follow a consistent layout: `src/` for source code, `tests/`, `docs/`, and `scripts/` for one-off utilities.
- **Documentation**: Create and maintain a `README` at all times.
- **Code Comments**: Make diligent use of docstring headers for all functions. Explain input parameters, function behavior, expected outputs, and error conditions.
- **Git Artifacts**: Create and maintain a `.gitignore`.
- **Python Version**: Use Python 3.13.7
