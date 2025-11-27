from GraphTsetlinMachine.graphs import Graphs
import numpy as np
from time import time
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine

class graph_tm:

    def __init__(self, args, number_of_nodes, node_names, games_train, games_test, edge_connections):
        self.args = args
        self.number_of_nodes = number_of_nodes
        self.node_names = node_names
        self.edge_connections = edge_connections
        self.games_train = games_train
        self.games_test = games_test

        # placeholders set up later
        self.graphs_train = None
        self.graphs_test = None
        self.Y_train = None
        self.Y_test = None
        self.tm = None
        
    '''
    Initializiation
    '''
    '''
    Creating the graphs for training
    '''
    def prepare_graphs(self):
        self.graphs_train = Graphs(
            self.args.number_of_graphs_train,
            #node_names=node_names,
            symbols=self.args.symbols,
            hypervector_size=self.args.hypervector_size,
            hypervector_bits=self.args.hypervector_bits,
            one_hot_encoding=self.args.one_hot_encoding
        )
        '''
        Creating the graphs for testing
        '''
        self.graphs_test = Graphs(
            self.args.number_of_graphs_test,
            #node_names=node_names,
            symbols=self.args.symbols,
            hypervector_size=self.args.hypervector_size,
            hypervector_bits=self.args.hypervector_bits,
            one_hot_encoding=self.args.one_hot_encoding
        )


    '''
    Adding a board_size * board_size number of nodes,
    to represent all locations on the board for both.
    '''
    def create_graphs_nodes(self, graphs, number_of_graphs, number_of_nodes):
        for graph_id in range(number_of_graphs):
            graphs.set_number_of_graph_nodes(graph_id, number_of_nodes)
        
        graphs.prepare_node_configuration()

        for graph_id in range(number_of_graphs):
            number_of_outgoing_edges = number_of_nodes - 1
            for node_name in self.node_names:
                graphs.add_graph_node(graph_id, node_name, number_of_outgoing_edges)


    '''
    Creating edges.
    Adding the edges to the nodes.
    Currently from every node an edge to all other nodes,
    because GraphTSM uses directional edges.
    Assigning the Edge type.
    IDEA:   It might be a bit two complex with bigger board sizes,
            so better use a edges inside a window size arround the node.
    '''
    def create_graphs_edges(self, graphs, number_of_graphs, number_of_nodes):
        graphs.prepare_edge_configuration()

        for graph_id in range(number_of_graphs):
            # WARNING: Plain might not be sufficient
            # IDEA: Maybe something like distance and direction would make more sense,
            # as highlight the relationship
            edge_type = "Plain"
            for node_name in self.node_names:
                for neighbor_node_name in self.node_names:
                    if node_name != neighbor_node_name:
                        graphs.add_graph_node_edge(graph_id, node_name, neighbor_node_name, edge_type)

    '''
    Creating neighbor edges.
    Connected every node with his 6 possible neighbors 
    Important in bigger board sizes.
    '''
    def create_graphs_neighbor_edges(self, graphs, number_of_graphs, number_of_nodes):
        graphs.prepare_edge_configuration()

        board_size = int(np.sqrt(number_of_nodes))
        
        if board_size * board_size != number_of_nodes:
            raise ValueError(f"Number of Nodes ({number_of_nodes}) isn't a perfect squarenumber. "
                             "Logic expects N*N-Board.")
        
        for graph_id in range(number_of_graphs):
            edge_type = "Plain"
            for node_name in self.node_names:
                try:
                    r_str, c_str = node_name.split(':')
                    r = int(r_str) # Row
                    c = int(c_str) # Column
                except ValueError:
                    print(f"Warning: Wrong nodename: {node_name}, contiuning.")
                    continue

                potential_neighbors_coords = [
                    (r-1, c),  
                    (r-1, c+1),
                    (r,   c-1),
                    (r,   c+1),
                    (r+1, c-1),
                    (r+1, c)
                ]

                for nr, nc in potential_neighbors_coords:
                    if (1 <= nr <= board_size) and (1 <= nc <= board_size):
                        neighbor_node_name = f"{nr}:{nc}"
                        graphs.add_graph_node_edge(graph_id, node_name, neighbor_node_name, edge_type)

    '''
    Creating neighbor edges up to 2nd degree (N1 + N2).
    Connected every node with his 6 direct neighbors (N1)
    and his 12 2-times neighbors (N2).
    '''
    def create_graphs_neighbor_2_edges(self, graphs, number_of_graphs, number_of_nodes):
        graphs.prepare_edge_configuration()

        board_size = int(np.sqrt(number_of_nodes))
        
        if board_size * board_size != number_of_nodes:
            raise ValueError(f"Number of Nodes ({number_of_nodes}) isn't a perfect squarenumber. "
                             "Logic expects N*N-Board.")
        
        n1_offsets = [
            (-1, 0),  
            (-1, 1), 
            ( 0,-1), 
            ( 0, 1), 
            ( 1,-1), 
            ( 1, 0)
        ]
        
        n2_offsets = [
            (-2, 0), (-2, 1), (-1, -1),
            (-2, 2), (-1, 2), ( 0, -2),
            ( 1,-2), ( 0, 2), ( 1, 1),
            ( 2,-2), ( 2,-1), ( 2, 0)
        ]

        for graph_id in range(number_of_graphs):
            for node_name in self.node_names:
                try:
                    r_str, c_str = node_name.split(':')
                    r = int(r_str) # Row
                    c = int(c_str) # Column
                except ValueError:
                    print(f"Warning: Wrong nodename: {node_name}, contiuning.")
                    continue

                # Gemini says that we should use different edge types, but i'm not sure if that is right.
                # edge_type = "Neighbor_N1" 

                # Edges for 1 grade
                for dr, dc in n1_offsets:
                    nr, nc = r + dr, c + dc
                    if (1 <= nr <= board_size) and (1 <= nc <= board_size):
                        neighbor_node_name = f"{nr}:{nc}"
                        graphs.add_graph_node_edge(graph_id, node_name, neighbor_node_name, "plain")

                # edges for 2 grade
                for dr, dc in n2_offsets:
                    nr, nc = r + dr, c + dc
                    if (1 <= nr <= board_size) and (1 <= nc <= board_size):
                        neighbor_node_name = f"{nr}:{nc}"
                        graphs.add_graph_node_edge(graph_id, node_name, neighbor_node_name, "plain")

    '''
    Load the learning data games into the different graphs, 
    by adding the board game data into the different nodes.
    Fixed sequence of data for better comparability between different learning runs in the future.
    '''
    def fill_graphs(self, graphs, number_of_graphs, games, Y_data):
        for graph_id in range(number_of_graphs):
            if not games:
                raise Exception('No games found')
            else:
                board, winner = games[graph_id % len(games)]

            for node_name in self.node_names:
                i_str, j_str = node_name.split(':')
                i = int(i_str) - 1
                j = int(j_str) - 1
                graphs.add_graph_node_property(graph_id, node_name, board[i][j])

            Y_data[graph_id] = np.uint32(winner)

        graphs.encode()

    '''
    Build the Tsetlin Machine with the given parameters.
    '''
    def build_tm(self):
        tm = MultiClassGraphTsetlinMachine(
            number_of_clauses = self.args.number_of_clauses,
            T = self.args.T,
            s = self.args.s,
            number_of_state_bits = self.args.number_of_state_bits,
            depth = self.args.depth,
            message_size = self.args.message_size,
            message_bits = self.args.message_bits,
            max_included_literals = self.args.max_included_literals,
            double_hashing = self.args.double_hashing,
            one_hot_encoding = self.args.one_hot_encoding,
            grid=(16*13,1,1),
            block=(128,1,1)   
        )
        return tm

    '''
    Running the full process of creating, training and testing the Graph Tsetlin Machine
    '''
    def run(self):

        '''
        Preparing the graphs for training and testing.
        '''
        self.prepare_graphs()

        '''
        Creating nodes for both training and testing graphs.
        '''
        self.create_graphs_nodes(self.graphs_train, self.args.number_of_graphs_train, self.number_of_nodes)
        self.create_graphs_nodes(self.graphs_test, self.args.number_of_graphs_test, self.number_of_nodes)

        '''
        Creating edges for both training and testing graphs.
        '''
        if self.args.edge_connections == "full":
            self.create_graphs_edges(self.graphs_train, self.args.number_of_graphs_train, self.number_of_nodes)
            self.create_graphs_edges(self.graphs_test, self.args.number_of_graphs_test, self.number_of_nodes)
        elif self.args.edge_connections == "neighbor":
            self.create_graphs_neighbor_edges(self.graphs_train, self.args.number_of_graphs_train, self.number_of_nodes)
            self.create_graphs_neighbor_edges(self.graphs_test, self.args.number_of_graphs_test, self.number_of_nodes)
        elif self.args.edge_connections == "neighbor_2":
            self.create_graphs_neighbor_2_edges(self.graphs_train, self.args.number_of_graphs_train, self.number_of_nodes)
            self.create_graphs_neighbor_2_edges(self.graphs_test, self.args.number_of_graphs_test, self.number_of_nodes)
        else:
            raise ValueError(f"Unknown edge_type: {self.args.edge_type}. Supported types are 'full' and 'neighbor'.")
        '''
        Filling the graphs with the board game data.
        '''
        self.Y_train = np.empty(self.args.number_of_graphs_train, dtype=np.uint32)
        self.Y_test = np.empty(self.args.number_of_graphs_test, dtype=np.uint32)

        self.fill_graphs(self.graphs_train, self.args.number_of_graphs_train, self.games_train, self.Y_train)
        self.fill_graphs(self.graphs_test, self.args.number_of_graphs_test, self.games_test, self.Y_test)

        '''
        Creating the Tsetlin Machine.
        '''
        self.tm = self.build_tm()

        '''
        Training and testing the Tsetlin Machine.
        '''
        return self.train_and_test()

    '''
    First do training.
    Second do testing.
    '''
    def train_and_test(self):
        results_train = []
        results_test = []
        start_time = time()
        for i in range(self.args.epochs):
            self.tm.fit(self.graphs_train, self.Y_train, epochs=1, incremental=True)

            result_test = 100*(self.tm.predict(self.graphs_test) == self.Y_test).mean()
            results_test.append(result_test)

            result_train = 100*(self.tm.predict(self.graphs_train) == self.Y_train).mean()
            results_train.append(result_train)

            print(f"[Epoch {i:03d}]  Training Accuracy: {result_train:.2f}  |  Test Accuracy: {result_test:.2f}")
        stop_time = time()
        time_taken = stop_time - start_time
        return results_train, results_test, time_taken

