"""Runs a lane of scripts/final/tasks_post.py with the machinery of scripts/final/orchestrate.py (resumable, retry once,
failures logged to stages/DECISIONS_LOG.md)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import orchestrate  # noqa: E402
import tasks_post  # noqa: E402

orchestrate.LANES = tasks_post.LANES
if __name__ == "__main__":
    orchestrate.main()
