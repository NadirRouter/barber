---
name: barber-stats
description: >
  Show what the barber hook actually removed from this session's tool output,
  replayed from the session log rather than estimated. Use when the user asks
  whether barber is working, what it saved, or runs /barber:stats.
---

The hook edits tool output before it reaches the model, so nothing in the
conversation shows it ran. This replays it over the real session log.

Find the current session's transcript, newest `.jsonl` under the project's log
directory (`~/.claude/projects/<cwd with / and . replaced by ->/`), then:

```bash
python3 "${CLAUDE_PLUGIN_ROOT}/contrib/replay_hook.py" 1 <transcript.jsonl>
```

Report the table as printed. Two things to say out loud, because both are
routinely misread:

- **Which denominator.** Only about half of tool-output tokens clear the
  TRIMMABLE / min-chars / JSON gates, so "% eligible" and "% all output" differ
  by roughly 2x and describe the same removal.
- **This is potential, not net.** A replay cannot model the agent re-running a
  command because the first result came back thinner.

If the numbers are near zero, that is usually the hook declining rather than
failing: short outputs, JSON bodies, and results from tools outside `TRIMMABLE`
are all left alone by design. `BARBER_HOOK_DISABLE=1` in the environment turns
it off entirely and is worth checking before debugging anything else.
