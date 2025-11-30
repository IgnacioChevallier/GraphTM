from pathlib import Path
from datetime import datetime, timezone
import json
import pickle

FILE_PATH_EXPLORATION = Path(__file__).parent / 'data' / 'exploration_results'
MODELS_DIR = Path(__file__).parent / 'models'


'''
Save all exploration results to `explorations.json`.
'''
def save_exploration_results(results):
    target_dir = FILE_PATH_EXPLORATION
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / "explorations.json"


    if isinstance(results, list):
        entries = list(results)
    else:
        entries = [results]

    normalized = []
    for entry in entries:
        try:
            data = dict(entry)
        except Exception:
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


def save_model_checkpoint(tm, test_accuracy, model_dir: Path | None = None, prefix: str = "tm_model", args=None):
    """
    Persist the trained TM using pickle so that the dashboard can inspect it.
    """
    if model_dir is None:
        model_dir = MODELS_DIR
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        accuracy_token = str(int(round(float(test_accuracy))))
    except Exception:
        accuracy_token = "unknown"

    timestamp = datetime.now(timezone.utc).strftime("%Y_%m_%d_%H_%M_%S")
    board_token = None
    if args is not None:
        board_token = getattr(args, "board_size", None) or getattr(args, "BOARD_SIZE", None)
    board_fragment = f"_board_{board_token}" if board_token is not None else ""
    filename = f"{prefix}_acc_{accuracy_token}{board_fragment}_date_{timestamp}.pkl"
    target_path = model_dir / filename

    state_dict = tm.save("")
    metadata = {
        "timestamp": timestamp,
        "test_accuracy": float(test_accuracy) if test_accuracy is not None else None,
    }
    if args is not None:
        try:
            metadata["args_snapshot"] = vars(args)
        except Exception:
            metadata["args_snapshot"] = str(args)

    state_dict.setdefault("metadata", {}).update(metadata)

    with open(target_path, "wb") as fh:
        pickle.dump(state_dict, fh)

    return target_path
        
