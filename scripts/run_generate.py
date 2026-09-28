"""Full-scale dataset generation driver."""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.seed.generator import generate  # noqa: E402

if __name__ == "__main__":
    t0 = time.perf_counter()
    frames = generate(to_disk=True)
    for name, frame in frames.items():
        print(f"{name:15s} rows={len(frame):9d} cols={frame.shape[1]}")
    print(f"elapsed {time.perf_counter() - t0:.1f}s")