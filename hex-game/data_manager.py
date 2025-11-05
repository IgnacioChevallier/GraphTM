import pickle
from pathlib import Path
import random

FILE_PATH_EXPLORATION = Path(__file__).parent / 'data' / 'exploration_results'


def save_exploration_results(file_path: Path | None, results: dict):
    target_dir = Path(file_path) if file_path is not None else FILE_PATH_EXPLORATION
    target_dir.mkdir(parents=True, exist_ok=True)

    # Random name to avoid overwriting
    out_path = target_dir / f"explored_tm_{random.randint(0,10**10)}.pkl"

    # args to dict
    data_to_save = dict(results)
    if "args" in data_to_save:
        try:
            data_to_save["args"] = vars(data_to_save["args"]) if hasattr(data_to_save["args"], "__dict__") or hasattr(data_to_save["args"], "__slots__") else data_to_save["args"]
        except Exception:
            pass

    # Add timestamp
    from datetime import datetime
    data_to_save.setdefault("timestamp", datetime.utcnow().isoformat() + "Z")

    with open(out_path, "wb") as fh:
        pickle.dump(data_to_save, fh)

