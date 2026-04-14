# Claude Instructions

## Project
ML model for target probability distribution recovery within the context of bitstrings with random noise; problem statement and approach detailed in proposal.md.

## My Preferences & Rules
- Always ask any clarifying questions
- For running scripts, always use python3 (not python)
- Never make design decisions without consulting me. If there are various options, present them clearly with tradeoffs.
- For large tasks, use sub-agents with cleanly divided and defined responsibilities.
- If you have any lingering uncertainties at any time, they must be stated explicitly.

## Code Rules
- Don't commit unless explicitly asked
- Less code is better than more code.
- Fallback mechanisms generally hide real failures and should be avoided unless explicitly necessary, in which case you must defend why that's the case.
- When writing tests, be especially sensitive to edge cases, and particularly those involving indexing.
- Avoid mixing scalar, slice, and list indices in a single numpy index expression (e.g. `a[scalar, :, list]`). This triggers advanced indexing broadcast rules and transposes the result unexpectedly. Use two-step indexing instead: `a[scalar][:, list]`.
- Use most significant bit first-ordering.
- When running pytest, always do so within a venv via source .venv/bin/activate. For example, a reasonable testing command would be: source .venv/bin/activate 2>/dev/null || true; which pytest; pytest tests/test_features.py -v 2>&1


## Style
- Concise responses — skip preamble, lead with the answer or the change
- No trailing summaries after edits unless the change is complex enough to warrant it
- No emojis
