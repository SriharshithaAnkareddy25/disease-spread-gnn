"""Run the same GRU-GNN training setup for each graph variant."""
import subprocess, sys
from pathlib import Path
root = Path(__file__).resolve().parents[2]
for graph in ["geographic", "gravity", "learned", "hybrid"]:
    print(f"\n===== Graph variant: {graph} =====", flush=True)
    subprocess.run([sys.executable, str(root/"src"/"training"/"train.py"), "--graph", graph], check=True, cwd=root)
