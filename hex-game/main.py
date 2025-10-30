from GraphTsetlinMachine.graphs import Graphs
import argparse
import numpy as np
import json
from pathlib import Path
from time import time
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine

# CONSTANTS
BOARD_SIZE = 3

'''
Overall arguments, that influence the final outcome of the GraphTM.
'''
def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=250, type=int) # Total number of times the model will iterate over the entire training dataset
    parser.add_argument("--number-of-clauses", default=20000, type=int) # Higher number = More complexity in the learned patters
    parser.add_argument("--T", default=25000, type=int) # Threshold for votes a clause needs
    parser.add_argument("--s", default=10.0, type=float) # Theshold to include literals
    parser.add_argument("--number-of-state-bits", default=8, type=int) # Depth 2^8 states
    parser.add_argument("--depth", default=1, type=int) # Message depth btw. nodes
    parser.add_argument("--symbols", nargs="+", default=['X', 'O', '.']) #Graph Symbols: X_Player1, O_Player2, ._Empty
    parser.add_argument("--hypervector-size", default=64, type=int) # Based on the number of symbols
    parser.add_argument("--hypervector-bits", default=2, type=int) # Bits represent the symbols (2 can represent 4 symbols)
    # Would not change, at least no change in most examples
    parser.add_argument("--message-size", default=256, type=int)
    parser.add_argument("--message-bits", default=2, type=int)
    parser.add_argument('--double-hashing', dest='double_hashing', default=False, action='store_true')
    parser.add_argument('--one-hot-encoding', dest='one_hot_encoding', default=False, action='store_true')
    
    parser.add_argument("--max-included-literals", default=32, type=int) # Max number of features learned per clause
    parser.add_argument("--number_of_graphs_train", default=20000, type=int) # Number of graphs used for training
    parser.add_argument("--number_of_graphs_test", default=2500, type=int) # Number of graphs used for testing

    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args

args = default_args()

'''
Create naming sceme for the nodes.
Calculate number of nodes.
'''
def initialize_hex_game(board_size):
    node_names = [f'{i}:{j}' for i in range(1, board_size+1) for j in range(1, board_size+1)]
    number_of_nodes = board_size * board_size
    return number_of_nodes, node_names

number_of_nodes, node_names  = initialize_hex_game(BOARD_SIZE)

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

print("Loading training and test data.")
dataset_file_train = Path(__file__).parent / 'data' / 'train_games.jsonl'
games_train = load_games_jsonl(dataset_file_train, args.number_of_graphs_train) # All games for the learning data
dataset_file_test = Path(__file__).parent / 'data' / 'test_games.jsonl'
games_test = load_games_jsonl(dataset_file_test, args.number_of_graphs_test) # All games for the test data

'''
Initializiation
'''
'''
Creating the graphs for training
'''
graphs_train = Graphs(
    args.number_of_graphs_train,
    #node_names=node_names,
    symbols=args.symbols,
    hypervector_size=args.hypervector_size,
    hypervector_bits=args.hypervector_bits,
    one_hot_encoding=args.one_hot_encoding
)
'''
Creating the graphs for testing
'''
graphs_test = Graphs(
    args.number_of_graphs_test,
    #node_names=node_names,
    symbols=args.symbols,
    hypervector_size=args.hypervector_size,
    hypervector_bits=args.hypervector_bits,
    one_hot_encoding=args.one_hot_encoding
)

'''
Adding a board_size * board_size number of nodes,
to represent all locations on the board for both.
'''
def create_graphs_nodes(graphs, number_of_graphs, number_of_nodes):
    for graph_id in range(number_of_graphs):
        graphs.set_number_of_graph_nodes(graph_id, number_of_nodes)
    
    graphs.prepare_node_configuration()

    for graph_id in range(number_of_graphs):
        number_of_outgoing_edges = number_of_nodes - 1
        for node_name in node_names:
            graphs.add_graph_node(graph_id, node_name, number_of_outgoing_edges)

print("Creating graph nodes.")
create_graphs_nodes(graphs_train, args.number_of_graphs_train, number_of_nodes)
create_graphs_nodes(graphs_test, args.number_of_graphs_test, number_of_nodes)

'''
Creating edges.
Adding the edges to the nodes.
Currently from every node an edge to all other nodes,
because GraphTSM uses directional edges.
Assigning the Edge type.
IDEA:   It might be a bit two complex with bigger board sizes,
        so better use a edges inside a window size arround the node.
'''
def create_graphs_edges(graphs, number_of_graphs, number_of_nodes):
    graphs.prepare_edge_configuration()

    for graph_id in range(number_of_graphs):
        # WARNING: Plain might not be sufficient
        # IDEA: Maybe something like distance and direction would make more sense,
        # as highlight the relationship
        edge_type = "Plain"
        for node_name in node_names:
            for neighbor_node_name in node_names:
                if node_name != neighbor_node_name:
                    graphs.add_graph_node_edge(graph_id, node_name, neighbor_node_name, edge_type)

print("Creating graph edges.")
create_graphs_edges(graphs_train, args.number_of_graphs_train, number_of_nodes)
create_graphs_edges(graphs_test, args.number_of_graphs_test, number_of_nodes)


Y_train = np.empty(args.number_of_graphs_train, dtype=np.uint32)
Y_test = np.empty(args.number_of_graphs_test, dtype=np.uint32)

'''
Load the learning data games into the different graphs, 
by adding the board game data into the different nodes.
Fixed sequence of data for better comparability between different learning runs in the future.
'''
def fill_graphs(graphs, number_of_graphs, games, Y_data):
    for graph_id in range(number_of_graphs):
        if not games:
            raise Exception('No games found')
        else:
            board, winner = games[graph_id % len(games)]

        for node_name in node_names:
            i_str, j_str = node_name.split(':')
            i = int(i_str) - 1
            j = int(j_str) - 1
            graphs.add_graph_node_property(graph_id, node_name, board[i][j])

        Y_data[graph_id] = np.uint32(winner)

    graphs.encode()

print("Fill the graphs.")
fill_graphs(graphs_train, args.number_of_graphs_train, games_train, Y_train)
fill_graphs(graphs_test, args.number_of_graphs_test, games_test, Y_test)

'''
Fill in all the data for the Multi Class Graph Tsetlin Machine
'''
tm = MultiClassGraphTsetlinMachine(
    number_of_clauses = args.number_of_clauses,
    T = args.T,
    s = args.s,
    number_of_state_bits = args.number_of_state_bits,
    depth = args.depth,
    message_size = args.message_size,
    message_bits = args.message_bits,
    max_included_literals = args.max_included_literals,
    double_hashing = args.double_hashing,
    one_hot_encoding = args.one_hot_encoding
)

'''
First do training.
Second do testing.
'''
for i in range(args.epochs):
    start_training = time()
    tm.fit(graphs_train, Y_train, epochs=1, incremental=True)
    stop_training = time()

    start_testing = time()
    result_test = 100*(tm.predict(graphs_test) == Y_test).mean()
    stop_testing = time()

    result_train = 100*(tm.predict(graphs_train) == Y_train).mean()
    
    print("%.2f %.2f %.2f %.2f" % (result_train, result_test, stop_training-start_training, stop_testing-start_testing))
'''
QUESTION: Does the rest do anything usefull?
'''
# weights = tm.get_state()[1].reshape(2, -1)
# for i in range(tm.number_of_clauses):
#         print("Clause #%d W:(%d %d)" % (i, weights[0,i], weights[1,i]), end=' ')
#         l = []
#         for k in range(graphs_train.hypervector_size * 2):
#             if tm.ta_action(0, i, k):
#                 if k < graphs_train.hypervector_size:
#                     l.append("x%d" % (k))
#                 else:
#                     l.append("NOT x%d" % (k - graphs_train.hypervector_size))

#         # for k in range(args.message_size * 2):
#         #     if tm.ta_action(1, i, k):
#         #         if k < args.message_size:
#         #             l.append("c%d" % (k))
#         #         else:
#         #             l.append("NOT c%d" % (k - args.message_size))

#         print(" AND ".join(l))

# #print(graphs_test.hypervectors)
# print(tm.hypervectors)
# #print(graphs_test.edge_type_id)
