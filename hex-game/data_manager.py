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


def _literal_name(idx: int, width: int, prefix: str) -> str:
    half = max(width // 2, 1)
    base_idx = idx if idx < half else idx - half
    label = f"{prefix}{base_idx}"
    return label if idx < half else f"NOT {label}"


def _clauses_for_depth(tm, depth: int, width: int, prefix: str) -> list[str]:
    clauses = []
    if width <= 0:
        return clauses

    for clause_idx in range(tm.number_of_clauses):
        literals = [
            _literal_name(literal_idx, width, prefix)
            for literal_idx in range(width)
            if tm.ta_action(depth, clause_idx, literal_idx)
        ]
        clauses.append(" AND ".join(literals) if literals else "<EMPTY>")
    return clauses


def _extract_clause_strings(tm) -> dict[str, list[str]]:
    clause_map: dict[str, list[str]] = {}

    literal_width = getattr(tm, "number_of_literals", 0)
    if literal_width:
        clause_map["depth0_literals"] = _clauses_for_depth(tm, depth=0, width=literal_width, prefix="X")

    message_width = getattr(tm, "number_of_message_literals", 0)
    for depth in range(1, getattr(tm, "depth", 1)):
        if message_width:
            key = f"depth{depth}_messages"
            clause_map[key] = _clauses_for_depth(tm, depth=depth, width=message_width, prefix=f"M{depth}_")

    return clause_map


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

    try:
        clause_strings = _extract_clause_strings(tm)
    except Exception as exc:  # pragma: no cover - best effort
        clause_strings = {"__error__": [str(exc)]}

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

    metadata["clauses"] = clause_strings

    state_dict.setdefault("metadata", {}).update(metadata)

    with open(target_path, "wb") as fh:
        pickle.dump(state_dict, fh)

    return target_path
        
