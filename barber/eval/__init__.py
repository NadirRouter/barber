"""barber.eval — the HotpotQA A/B harness that validated barber's defaults.

Requires the eval extras: ``pip install barber-llm[eval]``. Run it via the
``barber-eval`` console script. The judge prompt ships in JUDGE_PROMPT.md next
to this file; the full methodology is in docs/methodology.md in the repo.

This package imports nothing heavy at import time; the harness module pulls in
``datasets``/``openai``/``tiktoken`` only when actually run.
"""
