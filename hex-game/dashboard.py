import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import pickle
import sys
import re  # Import regular expression module
from pathlib import Path

'''
command to run the app:
streamlit run hex-game/dashboard.py
'''

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
# DATA LOADING FUNCTION
# -------------------------------------------------------------------
@st.cache_data
def load_model_data(model_path: Path):
    '''
    Loads the selected .pkl model file from the given path.
    It extracts the parameters and processes the raw model state
    (ta_state, clause_weights) into a structured Pandas DataFrame.
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
    Iterates through the model dictionary and pulls out simple data types
    (int, float, str, etc.) to be displayed in the sidebar.
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
    This is the core logic to interpret the model's internal state.
    '''
    try:
        '''
        Reading model metadata (dimensions).
        '''
        num_clauses = model_dict['number_of_clauses']
        num_outputs = model_dict['number_of_outputs']
        max_literals_storage = model_dict['max_included_literals']
        
        '''
        Processing 'ta_state' to count literals per clause.
        'ta_state' is assumed to be a flat array that needs reshaping.
        We count non-zero entries per clause row to find the literal count.
        '''
        ta_state_flat = model_dict['ta_state']
        ta_state_reshaped = ta_state_flat.reshape((num_clauses, max_literals_storage))
        literal_counts = np.count_nonzero(ta_state_reshaped, axis=1)

        '''
        Processing 'clause_weights' to determine clause relevance.
        We reshape the flat array and select the weights for the first output
        as the primary "relevance score".
        '''
        clause_weights_flat = model_dict['clause_weights']
        clause_weights_reshaped = clause_weights_flat.reshape((num_clauses, num_outputs))
        relevance_scores = clause_weights_reshaped[:, 0] # Use Output 0

    except KeyError as e:
        st.error(f"Error: Expected key {e} not found in model dictionary.")
        return None, None
    except Exception as e:
        st.error(f"Error analyzing model structure (ta_state/clause_weights): {e}")
        return None, None

    '''
    Block 3: Create the final DataFrame.
    This DataFrame holds all processed data for plotting.
    '''
    df_clauses = pd.DataFrame({
        "clause_id": [f"Clause_{i}" for i in range(num_clauses)],
        "literal_count": literal_counts,
        "relevance_score": relevance_scores
    })
    
    st.success(f"Model {model_path.name} loaded successfully.")
    return parameters, df_clauses

# -------------------------------------------------------------------
# HELPER FUNCTIONS
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

def parse_accuracy_from_filename(filename: str):
    '''
    Uses regular expressions to find an accuracy pattern (e.g., '_acc_XX')
    in the model's filename. Returns the integer value or None.
    '''
    match = re.search(r"_acc_(\d+)", filename)
    if match:
        try:
            return int(match.group(1)) # Return the captured number
        except ValueError:
            return None
    return None

# -------------------------------------------------------------------
# --- Streamlit App Layout ---
# -------------------------------------------------------------------

'''
Setting the page configuration to use a wide layout by default.
'''
st.set_page_config(layout="wide")
st.title("GraphTM: Graphical Model Overview")

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
    
    '''
    Parse the selected filename for its accuracy.
    '''
    model_accuracy = parse_accuracy_from_filename(selected_model_name)

    '''
    Load the data for the selected model.
    This uses the cached 'load_model_data' function.
    '''
    params, df_clauses = load_model_data(selected_model_path)

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

        # --- Main Dashboard ---
        st.header(f"Analysis for: {selected_model_name}")
        
        '''
        Section 1: Clause Structure Analysis (Literals)
        This section addresses the requests for literal distribution and max literals.
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
            (Issue Requirement 2)
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
            (Issue Requirement 3 & Accuracy)
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
        This section addresses the request for visualizing the most relevant clauses.
        (Issue Requirement 4)
        '''
        st.subheader("Clause Relevance Analysis (Weights)")
        st.info("This shows relevance based on clause weights (for Output 0).")
        
        df_clauses_sorted = df_clauses.sort_values(by="relevance_score", ascending=False)
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
        st.dataframe(df_clauses_sorted.head(top_n))

    else:
        st.error(f"Dashboard could not be loaded for {selected_model_name}.")