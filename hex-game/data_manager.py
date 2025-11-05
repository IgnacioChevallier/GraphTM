from pathlib import Path
from datetime import datetime
import json

FILE_PATH_EXPLORATION = Path(__file__).parent / 'data' / 'exploration_results'


def save_exploration_results(results):
    """Save one or more exploration result dicts into `explorations.json`.

    `results` may be a single dict or a list of dicts. Each dict may contain an
    `args` value that is an argparse.Namespace; this will be converted to a dict.
    """
    target_dir = FILE_PATH_EXPLORATION
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / "explorations.json"

    # Normalize to a list of dicts
    if isinstance(results, list):
        entries = list(results)
    else:
        entries = [results]

    normalized = []
    for entry in entries:
        try:
            data = dict(entry)
        except Exception:
            # If entry isn't dict-like, stringify it
            normalized.append({"value": str(entry), "timestamp": datetime.utcnow().isoformat() + "Z"})
            continue

        if "args" in data:
            try:
                args = data["args"]
                if hasattr(args, "__dict__") or hasattr(args, "__slots__"):
                    data["args"] = vars(args)
            except Exception:
                pass

        data.setdefault("timestamp", datetime.utcnow().isoformat() + "Z")
        normalized.append(data)

    # Load existing file (list) if present
    existing = []
    if out_path.exists():
        try:
            with open(out_path, "r", encoding="utf-8") as fh:
                existing = json.load(fh)
                if not isinstance(existing, list):
                    existing = [existing]
        except Exception:
            existing = []

    existing.extend(normalized)
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
        
