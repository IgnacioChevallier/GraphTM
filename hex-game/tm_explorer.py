import setup_game
import argparse
from itertools import product
import random
import data_manager
import os
import traceback

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

    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args

def new_exploration_args(current_index, permutate_exploration_params: bool = True):
    # Define the exploration options
    exploration_options = {
        "number_of_clauses": [10, 100, 500, 1000, 2000, 5000, 10000, 20000],
        "s": [0.5, 2.0, 5.0, 10.0, 15.0],
        "T": [1000, 5000, 10000, 20000],
        "number_of_state_bits": [4, 6, 8, 10],
        "epochs": [100]
    }

    # Build all combinations (Cartesian product) in a stable key order
    keys = list(exploration_options.keys())
    all_combinations = list(product(*(exploration_options[k] for k in keys)))

    if permutate_exploration_params:
        # Deterministic shuffle based on current_index so runs are reproducible
        rnd = random.Random(current_index)
        rnd.shuffle(all_combinations)

    if not all_combinations:
        raise ValueError("No exploration parameters available.")

    # Wrap index to available combinations
    idx = current_index % len(all_combinations)
    chosen_combo = all_combinations[idx]

    # Map back to dict of argument overrides (matches argparse attribute names)
    exploration_params = {k: v for k, v in zip(keys, chosen_combo)}

    # Build an argparse.Namespace with default values and apply the exploration overrides
    args = default_args()
    for key, value in exploration_params.items():
        if key in args.__dict__:
            setattr(args, key, value)

    return args


import concurrent.futures

import tempfile


def _pycuda_init_worker():
    """Initializer run once in each worker process to set a unique
    PYCUDA cache directory so parallel compilation doesn't corrupt a shared cache.
    """
    try:
        pid = os.getpid()
        cache_dir = tempfile.mkdtemp(prefix=f"pycuda_cache_{pid}_")
        # Set before any pycuda usage in this process
        os.environ["PYCUDA_CACHE_DIR"] = cache_dir
    except Exception:
        # Best-effort; if this fails, worker will proceed with default cache
        pass

def _explore_worker(exploration_index, starting_exploration_index, number_of_nodes, node_names, games_train, games_test):
    # compute global index used to deterministically pick params
    global_index = starting_exploration_index + exploration_index
    args = new_exploration_args(global_index)
    try:
        # Import graph_tm here so the PYCUDA_CACHE_DIR set by the worker initializer
        # is visible when any pycuda/GraphTsetlinMachine compilation happens.
        import graph_tm

        tm_instance = graph_tm.graph_tm(
            args,
            number_of_nodes,
            node_names,
            games_train,
            games_test
        )
        results_train, results_test, time_taken = tm_instance.run()
        results_payload = {
            "args": args,
            "results_train": results_train,
            "results_test": results_test,
            "time_taken": time_taken,
            "exploration_index": exploration_index,
        }
        out_path = data_manager.save_exploration_results(None, results_payload)
        return {"status": "ok", "exploration_index": exploration_index, "args": args,
                "results_train": results_train, "results_test": results_test,
                "time_taken": time_taken, "out_path": out_path}
    except Exception:
        return {"status": "error", "exploration_index": exploration_index, "traceback": traceback.format_exc()}

def explore_tms(starting_exploration_index, total_explorations, number_of_nodes, node_names, games_train, games_test, max_workers=None, use_processes=True):
    """
    Run explorations in parallel.
    - max_workers: number of parallel workers (defaults to min(total_explorations, cpu_count()))
    - use_processes: if True use ProcessPoolExecutor (better for CPU-bound); otherwise ThreadPoolExecutor.
    """
    if total_explorations <= 0:
        return

    if max_workers is None:
        try:
            cpu = os.cpu_count() or 1
        except Exception:
            cpu = 1
        max_workers = min(total_explorations, cpu)

    if use_processes:
        Executor = concurrent.futures.ProcessPoolExecutor
        executor_kwargs = {"max_workers": max_workers, "initializer": _pycuda_init_worker}
    else:
        Executor = concurrent.futures.ThreadPoolExecutor
        executor_kwargs = {"max_workers": max_workers}

    # submit all tasks
    with Executor(**executor_kwargs) as ex:
        futures = [
            ex.submit(
                _explore_worker,
                i,
                starting_exploration_index,
                number_of_nodes,
                node_names,
                games_train,
                games_test
            )
            for i in range(total_explorations)
        ]

        # iterate as results come in
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            if res.get("status") == "ok":
                print("Exploration Parameters:", res["args"])
                print("Training Results:", res["results_train"][-1])
                print("Testing Results:", res["results_test"][-1])
                print("Time Taken:", res["time_taken"])
                print(f"Saved exploration results to: {res.get('out_path')}")
            else:
                print(f"Exploration {res.get('exploration_index')} failed:")
                print(res.get("traceback"))


'''
Single run of the Graph Tsetlin Machine with given parameters.
'''
def run_single_tm(args, number_of_nodes, node_names, games_train, games_test):
    # import here to avoid triggering pycuda compilation before any worker initializer
    import graph_tm

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
        explore_tms(random.randint(0,10**10), 20, number_of_nodes, node_names, games_train, games_test)

if __name__ == "__main__":
    main()