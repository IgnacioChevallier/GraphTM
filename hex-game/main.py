from GraphTsetlinMachine.graphs import Graphs
import argparse
import numpy as np

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
    symbols = ['White', 'Black', 'Empty']
    number_of_nodes = board_size * board_size
    return number_of_nodes, node_names, symbols

# CONSTANTS
BOARD_SIZE = 11

number_of_nodes, node_names, symbols = initialize_hex_game(BOARD_SIZE);

print(node_names)
print(symbols)
print(number_of_nodes)

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
for graph_id in range(args.number_of_examples):
    x1 = random.choice(['A', 'B'])
    x2 = random.choice(['A', 'B'])
 
    graphs_train.add_graph_node_property(graph_id, 'Node 1', x1)
    graphs_train.add_graph_node_property(graph_id, 'Node 2', x2)

    if x1 == x2:
        Y_train[graph_id] = 0
    else:
        Y_train[graph_id] = 1

    if np.random.rand() <= args.noise:
        Y_train[graph_id] = 1 - Y_train[graph_id]

graphs_train.encode()