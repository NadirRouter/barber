"""The shell filter must save context without changing useful source."""
import subprocess
import sys


def test_plain_text_filter():
    source = "\n\n".join(
        f"Section {i} describes request routing for cluster node {i} in detail."
        for i in range(30)
    )
    source += "\n\nRetry handler: retry failed requests with exponential backoff."
    result = subprocess.run(
        [sys.executable, "-m", "barber.cli", "--query", "Where is the retry handler?"],
        input=source, text=True, capture_output=True, check=True,
    )
    assert "Retry handler:" in result.stdout
    assert len(result.stdout) < len(source)
    assert result.stderr == ""

    short = subprocess.run(
        [sys.executable, "-m", "barber.cli", "--query", "retry"],
        input="one short result\n", text=True, capture_output=True, check=True,
    )
    assert short.stdout == "one short result\n"

    costly = "\n\n".join(
        f"row {i} alpha beta" if i % 2 else f"row {i} refund policy detail here"
        for i in range(40)
    )
    unchanged = subprocess.run(
        [sys.executable, "-m", "barber.cli", "--query", "refund policy?",
         "--keep", "0.5"],
        input=costly, text=True, capture_output=True, check=True,
    )
    assert unchanged.stdout == costly

    bad_keep = subprocess.run(
        [sys.executable, "-m", "barber.cli", "--query", "retry",
         "--keep", "nan"],
        input=source, text=True, capture_output=True,
    )
    assert bad_keep.returncode == 2
