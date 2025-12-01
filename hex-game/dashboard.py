import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import pickle
import sys
import re  # Import regular expression module
from pathlib import Path


# command to run the app:
# streamlit run hex-game/dashboard.py

'''
Setting a higher recursion depth for loading complex pickle files
that might be deeply nested.
'''
sys.setrecursionlimit(50000)

'''
Defining the relative path to the models directory.
'''
MODEL_DIR = Path("hex-game/models")


# -------------------------------------------------------------------
# HELPER FUNCTIONS (ERWEITERT)
# -------------------------------------------------------------------

@st.cache_data
def find_model_files(model_directory: Path):
    '''
    Scans the provided directory for all files ending with the .pkl extension.
    Returns a list of Path objects.
    '''
    if not model_directory.is_dir():
        return []
    model_files = list(model_directory.glob("*.pkl"))
    return model_files

def generate_hex_node_names(board_size):
    """
    Generiert Knotennamen (z.B. '00', '01', ..., '22' für ein 3x3-Board).
    """
    if board_size < 1:
        return []
    # Die Knoten werden reihenweise von 0,0 bis N-1,N-1 benannt.
    names = [f"{r}{c}" for r in range(board_size) for c in range(board_size)]
    return names

def parse_metrics_from_filename(filename: str):
    '''
    Verwendet reguläre Ausdrücke, um Genauigkeit (accuracy) und 
    Boardgröße (board_size) aus dem Dateinamen zu extrahieren.
    '''
    match_acc = re.search(r"_acc_(\d+)", filename)
    match_board = re.search(r"_board_(\d+)", filename)
    
    accuracy = int(match_acc.group(1)) if match_acc else None
    board_size = int(match_board.group(1)) if match_board else None
    
    return accuracy, board_size


# -------------------------------------------------------------------
# DATA LOADING FUNCTION (ERWEITERT UM DEKODIERUNG)
# -------------------------------------------------------------------
@st.cache_data
def load_model_data(model_path: Path, board_size: int):
    '''
    Loads the selected .pkl model file from the given path.
    It extracts the parameters, processes the raw model state,
    and decodes the Tsetlin Clauses.
    '''

    st.info(f"Loading model: {model_path.name}")
    
    model_dict = None
    try:
        with open(model_path, 'rb') as f:
            model_dict = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading pickle file: {e}")
        return None, None

    '''
    Block 1: Extract simple model parameters.
    '''
    parameters = {}
    simple_types = (int, float, str, bool, tuple, np.uint32)
    for key, value in model_dict.items():
        if isinstance(value, simple_types):
            parameters[key] = value
        elif key == 's' and isinstance(value, tuple):
             parameters[key] = value

    '''
    Block 2: Extract and process clause data.
    '''
    try:
        '''
        Reading model metadata (dimensions).
        '''
        num_clauses = model_dict['number_of_clauses']
        num_outputs = model_dict['number_of_outputs']
        # max_literals_storage = model_dict['max_included_literals'] 
        
        hypervector_size = 32
        number_of_state_bits = model_dict['number_of_state_bits']
        
        '''
        Processing 'ta_state' to count literals per clause.
        '''
        ta_state_flat = model_dict['ta_state']
        if ta_state_flat.size % num_clauses != 0:
            raise ValueError(
                f"ta_state size {ta_state_flat.size} not divisible by number_of_clauses={num_clauses}"
            )
        ta_state_width = ta_state_flat.size // num_clauses
        ta_state_reshaped = ta_state_flat.reshape((num_clauses, ta_state_width))
        literal_counts = np.count_nonzero(ta_state_reshaped, axis=1)

        '''
        Processing 'clause_weights' to determine clause relevance.
        '''
        clause_weights_flat = model_dict['clause_weights']
        clause_weights_reshaped = clause_weights_flat.reshape((num_clauses, num_outputs))
        relevance_scores = clause_weights_reshaped[:, 0] # Use Output 0
        
        node_names = generate_hex_node_names(board_size)
        H = hypervector_size 
        S = number_of_state_bits
        num_nodes = len(node_names)
        
        total_node_literals_configured = 2 * num_nodes * H 
        
        decoded_clauses_list = []
        clause_states = ta_state_reshaped

        for clause_idx in range(num_clauses):
            literals = []
            
            for k in range(total_node_literals_configured):
                
                block_idx = k // 32
                action_bit_index_2d = block_idx * S + (S - 1)
                
                if action_bit_index_2d >= ta_state_width:
                    break 

                action_bit_value = clause_states[clause_idx, action_bit_index_2d]
                bit_in_block = k % 32
                
                if (action_bit_value & (1 << bit_in_block)) > 0:
                    node_idx = k // (2 * H) 
                    
                    if node_idx >= num_nodes:
                        break

                    node_name = node_names[node_idx]
                    literal_set_offset = k % (2 * H)

                    if literal_set_offset < H:
                        feature_idx = literal_set_offset
                        literal_str = f"{node_name}.x{feature_idx}"
                    else:
                        feature_idx = literal_set_offset - H
                        literal_str = f"~{node_name}.x{feature_idx}" 
                        
                    literals.append(literal_str)
            
            decoded_clauses_list.append(" AND ".join(literals) if literals else "(Empty Clause)")


    except KeyError as e:
        st.error(f"Error: Expected key {e} not found in model dictionary.")
        return None, None
    except Exception as e:
        st.error(f"Error analyzing model structure (ta_state/clause_weights): {e}")
        return None, None

    '''
    Block 3: Create the final DataFrame.
    '''
    df_clauses = pd.DataFrame({
        "clause_id": [f"Clause_{i}" for i in range(num_clauses)],
        "literal_count": literal_counts,
        "relevance_score": relevance_scores,
        "decoded_literals": decoded_clauses_list
    })
    
    st.success(f"Model {model_path.name} loaded successfully.")
    return parameters, df_clauses



'''
Setting the page configuration to use a wide layout by default.
'''
st.set_page_config(layout="wide")
st.title("GraphTM: Graphical Model Overview (Hex Game)")

'''
Setting up the Sidebar for model selection.
'''
st.sidebar.header("Model Control")

model_files_paths = find_model_files(MODEL_DIR)

if not model_files_paths:
    '''
    Display an error if the specified model directory is empty or missing.
    '''
    st.sidebar.error(f"No `.pkl` models found in folder `{MODEL_DIR}`.")
    st.error(f"Please ensure the folder `{MODEL_DIR}` exists and contains models.")
else:
    '''
    If models are found, create the dropdown selector.
    '''
    model_file_names = [path.name for path in model_files_paths]
    
    selected_model_name = st.sidebar.selectbox(
        "Select a model to analyze:",
        options=model_file_names,
        index=0
    )
    
    selected_model_path = MODEL_DIR / selected_model_name
    
    model_accuracy, board_size_from_filename = parse_metrics_from_filename(selected_model_name)
    
    board_size = board_size_from_filename
    if board_size is None:
        st.sidebar.warning("Board size not found in filename. Please set manually.")
        board_size = st.sidebar.number_input(
            "Board Size (N x N)", 
            min_value=3, max_value=10, value=3, step=1
        )
    else:
        st.sidebar.success(f"Board Size detected from filename: {board_size}x{board_size}")

    '''
    Load the data for the selected model (Boardgröße übergeben).
    '''
    params, df_clauses = load_model_data(selected_model_path, board_size)

    '''
    Display the main dashboard only if the model data
    was successfully loaded.
    '''
    if params is not None and df_clauses is not None:

        '''
        Display the extracted model parameters in the sidebar.
        '''
        st.sidebar.header("Model Parameters")
        st.sidebar.json(params)

        st.header(f"Analysis for: {selected_model_name}")
        
        df_clauses_sorted = df_clauses.sort_values(by="relevance_score", ascending=False)


        # --- Section 0: Decoded Clauses (Literal Interpretation) - NEU ---
        st.subheader("Decoded Clauses (Raw Literal Interpretation)")
        st.markdown(f"**Interpreted Board Size:** {board_size}x{board_size} (Nodes: {board_size**2})")
        st.info("This table shows the decoded literals of the most heavily weighted clauses. The interpretation is based on direct bit localisation in the state memory.")

        top_n_for_literals = st.slider(
            "Number of clauses to display literals for", 
            min_value=1, max_value=min(20, len(df_clauses_sorted)), value=5, 
            key="literal_display_slider"
        )
        
        # Anzeige der dekodierten Klauseln in einer Tabelle
        st.dataframe(
            df_clauses_sorted[['clause_id', 'relevance_score', 'literal_count', 'decoded_literals']].head(top_n_for_literals),
            column_order=('clause_id', 'relevance_score', 'literal_count', 'decoded_literals'),
            hide_index=True,
            use_container_width=True
        )
        # --- ENDE Section 0 ---


        '''
        Section 1: Clause Structure Analysis (Literals)
        '''
        st.subheader("Clause Structure Analysis (Literals)")
        st.info("""
        **Overfitting Analysis (Literals per Clause):**
        This chart shows how many clauses use a specific number of literals (based on `ta_state` array counts).
        A "long tail" (many clauses with a high number of literals) *could* be an indicator of overfitting.
        """)

        col1, col2 = st.columns([3, 1])

        with col1:
            '''
            Displaying the Histogram for literal distribution.
            '''
            max_bins = df_clauses["literal_count"].max() + 1
            fig_hist = px.histogram(
                df_clauses, 
                x="literal_count", 
                nbins=int(max_bins) if max_bins > 0 else 1, 
                title="Distribution of Literals per Clause"
            )
            fig_hist.update_layout(bargap=0.1, xaxis_title="Number of Literals")
            st.plotly_chart(fig_hist, use_container_width=True)

        with col2:
            '''
            Displaying key metrics: Accuracy, Max Literals, Avg Literals, Total Clauses.
            '''
            if model_accuracy is not None:
                st.metric(label="Model Accuracy", value=f"{model_accuracy}%")
            else:
                st.metric(label="Model Accuracy", value="N/A (Not in filename)")

            max_lit = df_clauses["literal_count"].max()
            st.metric(label="Maximum Literals Used", value=int(max_lit))
            
            avg_lit = df_clauses["literal_count"].mean()
            st.metric(label="Avg. Literals per Clause", value=f"{avg_lit:.2f}")
            
            total_clauses = len(df_clauses)
            st.metric(label="Total Clauses", value=int(total_clauses))

        '''
        Section 2: Clause Relevance Analysis (Weights)
        '''
        st.subheader("Clause Relevance Analysis (Weights)")
        st.info("This shows relevance based on clause weights (for Output 0).")
        
        top_n = st.slider("Number of most relevant clauses (Top-N)", min_value=5, max_value=100, value=20, key=selected_model_name)

        '''
        Displaying the Bar Chart for Top-N most relevant clauses.
        '''
        fig_bar = px.bar(
            df_clauses_sorted.head(top_n), 
            x="clause_id", 
            y="relevance_score", 
            title=f"Top {top_n} Most Relevant Clauses",
            hover_data=["literal_count"] 
        )
        fig_bar.update_layout(xaxis_title="Clause ID", yaxis_title="Relevance Score (Weight Output 0)")
        st.plotly_chart(fig_bar, use_container_width=True)

        '''
        Displaying the raw data for the Top-N clauses in a table.
        '''
        st.subheader(f"Raw Data for Top {top_n} Clauses")
        st.dataframe(df_clauses_sorted[['clause_id', 'relevance_score', 'literal_count', 'decoded_literals']].head(top_n))

    else:
        st.error(f"Dashboard could not be loaded for {selected_model_name}.")