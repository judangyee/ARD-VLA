# CLAUDE.md

## Ponytail rules

- Default mode for coding tasks (writing/adding/refactoring/fixing/reviewing/designing code): the laziest solution that actually works. Levels: lite/full(default)/ultra.
- Climb the ladder, stop at the first rung that holds: (1) does this need to exist at all (YAGNI)? (2) reuse what's already in the codebase (3) stdlib (4) native platform feature (5) already-installed dependency (6) one line (7) only then, minimal new code.
- Read the task and trace the real flow before picking a rung — laziness never skips understanding.
- Bug fix = root cause (grep all callers), not a patch on just the reported path.
- No unrequested abstractions, no boilerplate "for later", fewest files, shortest diff.
- Mark deliberate corner-cuts with a `ponytail:` comment naming the ceiling and upgrade path.
- Never simplify away input validation at trust boundaries, error handling against data loss, security, accessibility, or anything explicitly requested.
- Non-trivial logic gets one small runnable check (assert/demo/test), not a framework.
- Output: code first, then at most three lines on what was skipped and when to add it — no essays.
- Full rules: `.claude/skills/ponytail/SKILL.md` (and the related `ponytail-review`/`ponytail-audit`/`ponytail-debt`/`ponytail-gain`/`ponytail-help` skills).
