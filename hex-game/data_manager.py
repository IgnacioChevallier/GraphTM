from pathlib import Path
from datetime import datetime
import json

FILE_PATH_EXPLORATION = Path(__file__).parent / 'data' / 'exploration_results'


def save_exploration_results(results):
    target_dir = FILE_PATH_EXPLORATION
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / f"explorations.json"

    # normalize result to a plain dict
    data = dict(results)
    if "args" in data:
        try:
            args = data["args"]
            if hasattr(args, "__dict__") or hasattr(args, "__slots__"):
                data["args"] = vars(args)
        except Exception:
            pass

    data.setdefault("timestamp", datetime.utcnow().isoformat() + "Z")

    existing = []
    if out_path.exists():
        try:
            with open(out_path, "r", encoding="utf-8") as fh:
                existing = json.load(fh)
                if not isinstance(existing, list):
                    existing = [existing]
        except Exception:
            existing = []

    existing.append(data)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(existing, fh, ensure_ascii=False, indent=2, default=str)


def load_exploration_results():
    path = FILE_PATH_EXPLORATION / "explorations.json"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as fh:
        try:
            data = json.load(fh)
            return data
        except Exception:
            return []
        
