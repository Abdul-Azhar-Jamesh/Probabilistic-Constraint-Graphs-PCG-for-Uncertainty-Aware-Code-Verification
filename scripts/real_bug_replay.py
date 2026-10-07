"""Compatibility entrypoint for the expanded published-bug benchmark."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.real_bug_benchmark import main  # noqa: E402

if __name__ == "__main__":
    main()
