import graph_tm
import setup_game
import argparse
from itertools import product
import random
import data_manager

'''
Overall arguments, that influence the final outcome of the GraphTM.
'''
def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=100, type=int) # Total number of times the model will iterate over the entire training dataset
    parser.add_argument("--number-of-clauses", default=10, type=int) # Higher number = More complexity in the learned patters
    parser.add_argument("--T", default=100, type=int) # Threshold for votes a clause needs
    parser.add_argument("--s", default=1.0, type=float) # Theshold to include literals
    parser.add_argument("--number-of-state-bits", default=8, type=int) # Depth 2^8 states
    parser.add_argument("--depth", default=2, type=int) # Message depth btw. nodes
    parser.add_argument("--symbols", nargs="+", default=['X', 'O', '.']) #Graph Symbols: X_Player1, O_Player2, ._Empty
    parser.add_argument("--hypervector-size", default=32, type=int) # Based on the number of symbols
    parser.add_argument("--hypervector-bits", default=2, type=int) # Bits represent the symbols (2 can represent 4 symbols)
    # Would not change, at least no change in most examples
    parser.add_argument("--message-size", default=256, type=int)
    parser.add_argument("--message-bits", default=2, type=int)
    parser.add_argument('--double-hashing', dest='double_hashing', default=False, action='store_true')
    parser.add_argument('--one-hot-encoding', dest='one_hot_encoding', default=False, action='store_true')
    
    parser.add_argument("--max-included-literals", default=32, type=int) # Max number of features learned per clause
    parser.add_argument("--number_of_graphs_train", default=10000, type=int) # Number of graphs used for training
    parser.add_argument("--number_of_graphs_test", default=2500, type=int) # Number of graphs used for testing

    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args

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
            games_test
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
        games_test
    )
    results_train, results_test, time_taken = tm_instance.run()
    print("Training Results:", results_train[-1])
    print("Testing Results:", results_test[-1])
    print("Time Taken:", time_taken)

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