from GraphTsetlinMachine.graphs import Graphs
import json
from pathlib import Path
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine

'''
Create naming sceme for the nodes.
Calculate number of nodes.
'''
def initialize_hex_game(board_size):
    node_names = [f'{i}:{j}' for i in range(1, board_size+1) for j in range(1, board_size+1)]
    number_of_nodes = board_size * board_size
    return number_of_nodes, node_names

'''
Load the prepared board games from the json.
'''
def load_games_jsonl(path: Path, limit: int | None = None):
    games = []
    if not path.exists():
        return games
    with path.open('r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                board = obj.get('board')
                winner = int(obj.get('winner'))
                games.append((board, winner))
            except Exception:
                continue
            if limit is not None and len(games) >= limit:
                break
    return games

def setup_game(args, BOARD_SIZE):
    print("Setting up the Hex game environment and loading data.")

    number_of_nodes, node_names  = initialize_hex_game(BOARD_SIZE)

    dataset_file_train = Path(__file__).parent / 'data' / 'train' / f'games_{BOARD_SIZE}x{BOARD_SIZE}.jsonl'
    games_train = load_games_jsonl(dataset_file_train, args.number_of_graphs_train) # All games for the learning data
    dataset_file_test = Path(__file__).parent / 'data' / 'test' / f'games_{BOARD_SIZE}x{BOARD_SIZE}.jsonl'
    games_test = load_games_jsonl(dataset_file_test, args.number_of_graphs_test) # All games for the test data

    return number_of_nodes, node_names, games_train, games_test