import graph_tm
import setup_game
import argparse
from pathlib import Path
from datetime import datetime

'''
Overall arguments, that influence the final outcome of the GraphTM.
'''
def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=100, type=int) # Total number of times the model will iterate over the entire training dataset
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
    parser.add_argument("--save-model", default="", type=str, help="Path to save the trained model. If empty, model is not saved. If 'auto', generates a timestamped filename.")

    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args

def calculate_exploration_parameters(args):
    exploration_params = []
    for clauses in args.explore_number_of_clauses:
        for s in args.explore_s:
            for T in args.explore_T:
                param_set = {
                    'number_of_clauses': clauses,
                    's': s,
                    'T': T
                }
                exploration_params.append(param_set)
    return exploration_params

def explore_tms(number_of_nodes, node_names, games_train, games_test):
    pass

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
    print("Training Results:", results_train)
    print("Testing Results:", results_test)
    print("Time Taken:", time_taken)
    
    # Save the model if requested
    if args.save_model:
        if args.save_model == "auto":
            # Generate timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_dir = Path(__file__).parent / "models"
            model_dir.mkdir(exist_ok=True)
            save_path = model_dir / f"tm_model_{timestamp}.pkl"
        else:
            save_path = Path(args.save_model)
            save_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving model to {save_path}")
        tm_instance.tm.save(str(save_path))
        print(f"Model saved successfully!")

'''
Main Function to start either single run or exploration.
'''
def main(single_run: bool = True, BOARD_SIZE: int = 3):
    number_of_nodes, node_names, games_train, games_test = setup_game.setup_game(default_args(), BOARD_SIZE)
    if single_run:
        run_single_tm(default_args(), number_of_nodes, node_names, games_train, games_test)
    else:
        explore_tms(number_of_nodes, node_names, games_train, games_test)

if __name__ == "__main__":
    main()