# Lessons

Patterns captured from user corrections, so the same mistake is not repeated.
Reviewed at the start of each session.

## Review scope

- When asked for a deep project review, cover the whole codebase and its operating
  assumptions. Do not narrow the review to recent changes unless explicitly asked.

## TODO presentation

- Keep completed tasks checked and strike through their full descriptions in
  `tasks/todo.md`. Also strike through historical deferred items when later passes
  complete them; leave unfinished portions unstruck.

## Environment

- When CI pins an exact Python patch release, verify the setup-python manifest
  supports the selected OS and architecture and pin the runner accordingly.
  A moving `ubuntu-latest` can select an OS newer than that Python build supports;
  local validation does not establish hosted-runner compatibility.

- **Use Python 3.13 even if the default `python3` is older.** This machine's default
  `python3` is 3.9.6; the project standard (CLAUDE.md) is 3.13. The 3.13 interpreter
  lives at `/opt/homebrew/bin/python3.13`. Always build `.venv` from it.

## Don't assert environment constraints without verifying — probe first

I claimed real Clay weights "can't be downloaded — no network" and leaned on it to
justify the offline reference backend. The user corrected it: the machine is
networked (we were chatting over it). A 30-second probe (`curl huggingface.co`,
check disk, `pip index versions torch`) would have shown it was reachable.

**Why it matters:** stating a false blocker steers the whole plan (and can quietly
lower the bar on what we deliver). The real blockers were different and more
interesting (the PRD's HF repo is fictional; Clay needs a non-trivial input
contract; `claymodel` won't install on 3.13).

**How to apply:** before citing a constraint (no network, no GPU, too big, won't
install), run a cheap read-only probe and state findings as evidence. If I can't
verify, say "let me check" — don't assert.

