from GraphTsetlinMachine.graphs import Graphs
import argparse
import numpy as np
import json
from pathlib import Path
from time import time
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine

def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--number-of-clauses", default=10, type=int)
    parser.add_argument("--T", default=100, type=int)
    parser.add_argument("--s", default=1.0, type=float)
    parser.add_argument("--number-of-state-bits", default=8, type=int)
    parser.add_argument("--depth", default=2, type=int)
    parser.add_argument("--hypervector-size", default=32, type=int)
    parser.add_argument("--hypervector-bits", default=2, type=int)
    parser.add_argument("--message-size", default=256, type=int)
    parser.add_argument("--message-bits", default=2, type=int)
    parser.add_argument('--double-hashing', dest='double_hashing', default=False, action='store_true')
    parser.add_argument('--one-hot-encoding', dest='one_hot_encoding', default=False, action='store_true')
    parser.add_argument("--noise", default=0.01, type=float)
    parser.add_argument("--number-of-examples", default=10000, type=int)
    parser.add_argument("--max-included-literals", default=4, type=int)

    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args

args = default_args()

def initialize_hex_game(board_size):
    node_names = [f'{i}:{j}' for i in range(1, board_size+1) for j in range(1, board_size+1)]
    symbols = ['X', 'O', '.']
    number_of_nodes = board_size * board_size
    return number_of_nodes, node_names, symbols

# CONSTANTS
BOARD_SIZE = 11

number_of_nodes, node_names, symbols = initialize_hex_game(BOARD_SIZE);

print(node_names)
print(symbols)
print(number_of_nodes)

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

graphs_train = Graphs(
    args.number_of_examples,
    node_names=node_names,
    symbols=symbols,
    hypervector_size=args.hypervector_size,
    hypervector_bits=args.hypervector_bits,
    one_hot_encoding=args.one_hot_encoding
)

for graph_id in range(args.number_of_examples):
    graphs_train.set_number_of_graph_nodes(graph_id, number_of_nodes)

graphs_train.prepare_node_configuration()

for graph_id in range(args.number_of_examples):
    number_of_outgoing_edges = number_of_nodes - 1;
    for node_name in node_names:
        graphs_train.add_graph_node(graph_id, node_name, number_of_outgoing_edges)

graphs_train.prepare_edge_configuration()

for graph_id in range(args.number_of_examples):
    edge_type = "Plain"
    for node_name in node_names:
        for neighbor_node_name in node_names:
            if node_name != neighbor_node_name:
                graphs_train.add_graph_node_edge(graph_id, node_name, neighbor_node_name, edge_type)

Y_train = np.empty(args.number_of_examples, dtype=np.uint32)

dataset_file = Path(__file__).parent / 'data' / 'games.jsonl'
games = load_games_jsonl(dataset_file)

for graph_id in range(args.number_of_examples):
    if not games:
        raise Exception('No games found')
    else:
        board, winner = games[graph_id % len(games)]

    for node_name in node_names:
        i_str, j_str = node_name.split(':')
        i = int(i_str) - 1
        j = int(j_str) - 1
        graphs_train.add_graph_node_property(graph_id, node_name, board[i][j])

    Y_train[graph_id] = np.uint32(winner)

graphs_train.encode() 


tm = MultiClassGraphTsetlinMachine(
    args.number_of_clauses,
    args.T,
    args.s,
    number_of_state_bits = args.number_of_state_bits,
    depth = args.depth,
    message_size = args.message_size,
    message_bits = args.message_bits,
    max_included_literals = args.max_included_literals,
    double_hashing = args.double_hashing,
    one_hot_encoding = args.one_hot_encoding
)

for i in range(args.epochs):
    start_training = time()
    tm.fit(graphs_train, Y_train, epochs=1, incremental=True)
    stop_training = time()

    start_testing = time()
    # result_test = 100*(tm.predict(graphs_test) == Y_test).mean()
    stop_testing = time()

    result_train = 100*(tm.predict(graphs_train) == Y_train).mean()

    #print("%d %.2f %.2f %.2f %.2f" % (i, result_train, result_test, stop_training-start_training, stop_testing-start_testing))
    print("%d %.2f %.2f %.2f" % (i, result_train, stop_training-start_training, stop_testing-start_testing))

weights = tm.get_state()[1].reshape(2, -1)
for i in range(tm.number_of_clauses):
        print("Clause #%d W:(%d %d)" % (i, weights[0,i], weights[1,i]), end=' ')
        l = []
        for k in range(graphs_train.hypervector_size * 2):
            if tm.ta_action(0, i, k):
                if k < graphs_train.hypervector_size:
                    l.append("x%d" % (k))
                else:
                    l.append("NOT x%d" % (k - graphs_train.hypervector_size))

        # for k in range(args.message_size * 2):
        #     if tm.ta_action(1, i, k):
        #         if k < args.message_size:
        #             l.append("c%d" % (k))
        #         else:
        #             l.append("NOT c%d" % (k - args.message_size))

        print(" AND ".join(l))

#print(graphs_test.hypervectors)
print(tm.hypervectors)
#print(graphs_test.edge_type_id)
