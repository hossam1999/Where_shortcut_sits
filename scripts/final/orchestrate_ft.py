"""Third lane: the four end-to-end fine-tuning tasks at the end of the trap lane (scripts/final/tasks.py), run early in
parallel. The trap lane checks done-markers before each task, so it will skip them when it arrives (hours later)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import orchestrate  # noqa: E402
import tasks  # noqa: E402

orchestrate.LANES = {"ft": [t for t in tasks.TRAPS if t["name"].startswith("ft_")]}
if __name__ == "__main__":
    orchestrate.main()
