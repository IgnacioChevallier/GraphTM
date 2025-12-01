import graph_tm
import setup_game
import argparse
from itertools import product
import random
import numpy as np
import data_manager

'''
Overall arguments, that influence the final outcome of the GraphTM.
'''
def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=3, type=int) # Total number of times the model will iterate over the entire training dataset
    parser.add_argument("--number-of-clauses", default=10, type=int) # Higher number = More complexity in the learned patters
    parser.add_argument("--T", default=10, type=int) # Threshold for votes a clause needs
    parser.add_argument("--s", default=0.5, type=float) # Theshold to include literals
    parser.add_argument("--number-of-state-bits", default=8, type=int) # Depth 2^8 states
    parser.add_argument("--depth", default=2, type=int) # Message depth btw. nodes
    parser.add_argument("--symbols", nargs="+", default=['X', 'O', '.']) #Graph Symbols: X_Player1, O_Player2, ._Empty
    parser.add_argument("--hypervector-size", default=16, type=int) # Based on the number of symbols
    parser.add_argument("--hypervector-bits", default=2, type=int) # Bits represent the symbols (2 can represent 4 symbols)
    # Would not change, at least no change in most examples
    parser.add_argument("--message-size", default=256, type=int)
    parser.add_argument("--message-bits", default=2, type=int)
    parser.add_argument('--double-hashing', dest='double_hashing', default=False, action='store_true')
    parser.add_argument('--one-hot-encoding', dest='one_hot_encoding', default=True, action='store_true')
    
    parser.add_argument("--max-included-literals", default=10, type=int) # Max number of features learned per clause
    parser.add_argument("--number_of_graphs_train", default=100000, type=int) # Number of graphs used for training
    parser.add_argument("--number_of_graphs_test", default=100000, type=int) # Number of graphs used for testing
    parser.add_argument("--edge-connections", default="full", type=str,
                    help="Type of edge connections: full, neighbor, or neighbor_2")

    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args


def print_graph_tm_clauses(tm, hypervector_size, node_names):
    
    H = hypervector_size
    num_nodes = len(node_names)
    S = tm.number_of_state_bits
    
    total_node_literals_configured = 2 * num_nodes * H 
    weights = tm.get_weights().transpose() 
    num_classes = weights.shape[1]
    
    clause_states = tm.get_ta_states(depth=0)
    
    if clause_states.ndim != 2:
        print(f"FATAL ERROR: Expected 2D array from tm.get_ta_states(0), but got {clause_states.ndim}D array.")
        return

    inner_dim_size = clause_states.shape[1] 
    num_blocks = inner_dim_size // S 
    max_literals_allocatable = num_blocks * 32
    
    literal_limit = min(total_node_literals_configured, max_literals_allocatable)
    
    print("\n--- Tsetlin Machine Clauses (Depth 0: Node Features) ---")
    
    if max_literals_allocatable < total_node_literals_configured:
        print(f"🛑 Warning: Only first {max_literals_allocatable} literals (of {total_node_literals_configured}) could be shown.")

    CLAUSE_LIMIT = 20
    clauses_to_inspect = min(CLAUSE_LIMIT, tm.number_of_clauses)

    for clause_idx in range(clauses_to_inspect):
        
        weights_str = " ".join([f"{weights[clause_idx, c]:>4d}" for c in range(num_classes)])
        weights_label = f"W:({weights_str})"
        
        literals = []
        for k in range(literal_limit):
            
            block_idx = k // 32
            action_bit_index_2d = block_idx * S + (S - 1) 
            
            action_bit_value = clause_states[clause_idx, action_bit_index_2d]
            bit_in_block = k % 32
            
            if (action_bit_value & (1 << bit_in_block)) > 0:
                
                node_idx = k // (2 * H) 
                node_name = node_names[node_idx]
                literal_set_offset = k % (2 * H)

                if literal_set_offset < H:
                    feature_idx = literal_set_offset
                    literal_str = f"{node_name}.x{feature_idx}"
                else:
                    feature_idx = literal_set_offset - H
                    literal_str = f"NOT {node_name}.x{feature_idx}"
                    
                literals.append(literal_str)

        clause_str = f"Clause #{clause_idx:<4d} {weights_label}: " + " AND ".join(literals)
        print(clause_str)

    print("------------------------------------------------------")



'''   
Based on the current index, generate a new set of exploration parameters.
Return the updated args.
'''
def new_exploration_args(current_index, permutate_exploration_params: bool = True):
    exploration_options = {
        "number_of_clauses": [10, 100, 500, 1000, 2000, 5000, 10000, 20000],
        "s": [0.5, 2.0, 5.0, 10.0, 15.0],
        "T": [1000, 5000, 10000, 20000],
        "number_of_state_bits": [4, 6, 8, 10],
        "number_of_graphs_train": [5000, 10000, 20000, 40000],
        "epochs": [50] # for now keeping epochs constant
    }

    '''
    Change default arguments to the new explore params.
    '''
    keys = list(exploration_options.keys())
    all_combinations = list(product(*(exploration_options[k] for k in keys)))

    if permutate_exploration_params:
        rnd = random.Random(current_index)
        rnd.shuffle(all_combinations)

    if not all_combinations:
        raise ValueError("No exploration parameters available.")

    idx = current_index % len(all_combinations)
    chosen_combo = all_combinations[idx]

    exploration_params = {k: v for k, v in zip(keys, chosen_combo)}

    args = default_args()
    for key, value in exploration_params.items():
        if key in args.__dict__:
            setattr(args, key, value)

    return args

'''
Run multiple explorations of the Graph Tsetlin Machine with different parameters.
Save the results in "data/exploration_results" after all explorations are done.
'''
def explore_tms(starting_exploration_index, total_explorations, number_of_nodes, node_names, games_train, games_test):
    total_exploration_results = []
    for i in range(total_explorations):
        args = new_exploration_args(starting_exploration_index + i)
        tm_instance = graph_tm.graph_tm(
            args,
            number_of_nodes,
            node_names,
            games_train,
            games_test,
            edge_connections="full"
        )
        results_train, results_test, time_taken = tm_instance.run()
        # print("Exploration Parameters:", args)
        # print("Training Results:", results_train[-1])
        # print("Testing Results:", results_test[-1])
        # print("Time Taken:", time_taken)

        results_payload = {
            "args": args,
            "results_train": results_train,
            "results_test": results_test,
            "time_taken": time_taken,
            "exploration_index": i,
        }

        total_exploration_results.append(results_payload)

    data_manager.save_exploration_results(total_exploration_results)


'''
Single run of the Graph Tsetlin Machine with given parameters.
'''
def run_single_tm(args, number_of_nodes, node_names, games_train, games_test):
    tm_instance = graph_tm.graph_tm(
        args,
        number_of_nodes,
        node_names,
        games_train,
        games_test,
        edge_connections="full"
    )
    results_train, results_test, time_taken = tm_instance.run()
    board_size = int(len(node_names) ** 0.5)
    print("Training Results:", results_train[-1])
    print("Testing Results:", results_test[-1])
    print("Time Taken:", time_taken)
    print(f"Board Size: {board_size} x {board_size}")
    print("Number of clauses:", tm_instance.tm.number_of_clauses)
    print_graph_tm_clauses(
            tm_instance.tm,          # Die Tsetlin-Maschine
            args.hypervector_size,   # Die Hypervektor-Größe aus den Programm-Argumenten
            node_names               # Die Liste der Knotennamen
        )


'''
Main Function to start either single run or exploration.
'''
def main(single_run: bool = True, BOARD_SIZE: int = 3):
    number_of_nodes, node_names, games_train, games_test = setup_game.setup_game(default_args(), BOARD_SIZE)
    if single_run:
        run_single_tm(default_args(), number_of_nodes, node_names, games_train, games_test)
    else:
        explore_tms(random.randint(0,10**10), 50, number_of_nodes, node_names, games_train, games_test)


if __name__ == "__main__":
    main()