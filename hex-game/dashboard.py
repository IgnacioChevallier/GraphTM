import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pickle
import sys
import re  # Import regular expression module
from pathlib import Path
import math


#command to run the app:
#streamlit run hex-game/dashboard.py

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
# HELPER FUNCTIONS
# -------------------------------------------------------------------

# Hex game symbols - order matters for hypervector mapping
HEX_SYMBOLS = ['X', 'O', '.']  # X=Player1(Black), O=Player2(White), .=Empty

def get_clause_literals(ta_state_row, number_of_state_bits=8):
    '''
    Extracts the active literals from a clause's TA state row.
    The ta_state is stored as packed 32-bit integers, where each literal
    uses 'number_of_state_bits' bits.
    
    Returns a list of tuples: (literal_index, ta_state_value)
    Only returns literals where the TA state indicates inclusion (>= threshold).
    '''
    active_literals = []
    literal_idx = 0
    
    # Calculate the threshold for inclusion (state >= 2^(bits-1) means included)
    threshold = 2 ** (number_of_state_bits - 1)
    mask = (1 << number_of_state_bits) - 1  # e.g., 0xFF for 8 bits
    
    # Number of literals packed into each 32-bit integer
    bits_per_int = 32
    literals_per_int = bits_per_int // number_of_state_bits
    
    for packed_value in ta_state_row:
        # Convert to unsigned 32-bit integer
        packed = int(packed_value) & 0xFFFFFFFF
        
        # Extract each literal's state from the packed value
        for i in range(literals_per_int):
            # Extract the state for this literal
            state = (packed >> (i * number_of_state_bits)) & mask
            
            # Check if literal is included (state >= threshold)
            if state >= threshold:
                active_literals.append((literal_idx, state))
            
            literal_idx += 1
    
    return active_literals


def interpret_literals_for_hex(active_literals, hypervectors, hypervector_size, board_size):
    '''
    Interprets the active literals in the context of a Hex game.
    
    The literals are hypervector bits. We need to map them back to symbols
    by checking which symbol hypervectors they match.
    
    For each literal index:
    - If index < hypervector_size: it's a positive literal (symbol present)
    - If index >= hypervector_size: it's a negative literal (symbol absent), index -= hypervector_size
    
    Returns a dict with:
    - 'positive_symbols': dict mapping symbol_idx -> list of matching literal indices
    - 'negative_symbols': dict mapping symbol_idx -> list of matching literal indices  
    - 'contradictions': list of symbols that have both positive and negative evidence
    '''
    num_symbols = min(len(HEX_SYMBOLS), len(hypervectors))
    
    # Build reverse mapping: hypervector_bit -> list of symbol indices that use this bit
    bit_to_symbols = {}
    for sym_idx in range(num_symbols):
        sym_hv = hypervectors[sym_idx]
        for bit_pos in sym_hv:
            bit_pos = int(bit_pos)  # Ensure it's an integer
            if bit_pos not in bit_to_symbols:
                bit_to_symbols[bit_pos] = []
            bit_to_symbols[bit_pos].append(sym_idx)
    
    # Count matches for each symbol (positive and negative)
    positive_counts = {i: 0 for i in range(num_symbols)}
    negative_counts = {i: 0 for i in range(num_symbols)}
    
    positive_bits = []
    negative_bits = []
    
    for lit_idx, ta_state in active_literals:
        if lit_idx < hypervector_size:
            # Positive literal
            positive_bits.append(lit_idx)
            if lit_idx in bit_to_symbols:
                for sym_idx in bit_to_symbols[lit_idx]:
                    if sym_idx < num_symbols:  # Safety check
                        positive_counts[sym_idx] += 1
        else:
            # Negative literal
            actual_bit = lit_idx - hypervector_size
            negative_bits.append(actual_bit)
            if actual_bit in bit_to_symbols:
                for sym_idx in bit_to_symbols[actual_bit]:
                    if sym_idx < num_symbols:  # Safety check
                        negative_counts[sym_idx] += 1
    
    # Determine which symbols are strongly indicated
    hypervector_bits = hypervectors.shape[1] if len(hypervectors.shape) > 1 else 1
    threshold = max(1, hypervector_bits // 2)  # At least half the bits should match, minimum 1
    
    result = {
        'positive_symbols': [],  # Symbols that should be present
        'negative_symbols': [],  # Symbols that should be absent  
        'positive_counts': positive_counts,
        'negative_counts': negative_counts,
        'contradictions': [],
        'positive_bits': positive_bits,
        'negative_bits': negative_bits,
    }
    
    for sym_idx in range(num_symbols):
        pos_strong = positive_counts[sym_idx] >= threshold
        neg_strong = negative_counts[sym_idx] >= threshold
        
        if pos_strong and neg_strong:
            result['contradictions'].append(sym_idx)
        elif pos_strong:
            result['positive_symbols'].append(sym_idx)
        elif neg_strong:
            result['negative_symbols'].append(sym_idx)
    
    return result


def create_hex_board_visualization(board_size, interpretation, title="Clause Pattern"):
    '''
    Creates a visual representation of what the clause is looking for on a Hex board.
    
    Uses Plotly to draw hexagonal cells with colors indicating:
    - Green: Symbol should be present (X, O, or .)
    - Red: Symbol should be absent (NOT X, NOT O, NOT .)
    - Yellow: Contradiction (both present and absent)
    - Gray: No constraint
    '''
    
    # Create figure
    fig = go.Figure()
    
    # Hexagon parameters
    hex_size = 1.0
    hex_height = hex_size * math.sqrt(3)
    
    # Symbol display
    symbol_names = {0: 'X (Black)', 1: 'O (White)', 2: '. (Empty)'}
    symbol_short = {0: 'X', 1: 'O', 2: '.'}
    
    # Prepare legend text
    legend_items = []
    
    if interpretation['positive_symbols']:
        pos_names = [symbol_names[s] for s in interpretation['positive_symbols']]
        legend_items.append(f"✅ Must have: {', '.join(pos_names)}")
    
    if interpretation['negative_symbols']:
        neg_names = [symbol_names[s] for s in interpretation['negative_symbols']]
        legend_items.append(f"❌ Must NOT have: {', '.join(neg_names)}")
    
    if interpretation['contradictions']:
        cont_names = [symbol_names[s] for s in interpretation['contradictions']]
        legend_items.append(f"⚠️ Contradictions: {', '.join(cont_names)}")
    
    # Draw board cells
    annotations = []
    
    for row in range(board_size):
        for col in range(board_size):
            # Hex offset positioning
            x = col * 1.5 * hex_size + row * 0.75 * hex_size
            y = row * hex_height * 0.5
            
            # Create hexagon vertices
            angles = np.linspace(0, 2*np.pi, 7)[:-1] + np.pi/6
            hex_x = x + hex_size * 0.5 * np.cos(angles)
            hex_y = y + hex_size * 0.5 * np.sin(angles)
            
            # Determine cell color based on interpretation
            cell_text = ""
            cell_color = "lightgray"
            
            # Build cell text showing constraints
            pos_syms = [symbol_short[s] for s in interpretation['positive_symbols']]
            neg_syms = [symbol_short[s] for s in interpretation['negative_symbols']]
            cont_syms = [symbol_short[s] for s in interpretation['contradictions']]
            
            if cont_syms:
                cell_color = "rgba(255, 200, 0, 0.7)"  # Yellow for contradictions
                cell_text = "⚠️"
            elif pos_syms and neg_syms:
                cell_color = "rgba(100, 200, 100, 0.5)"  # Light green
                cell_text = f"{'/'.join(pos_syms)}"
            elif pos_syms:
                cell_color = "rgba(100, 200, 100, 0.7)"  # Green
                cell_text = f"{'/'.join(pos_syms)}"
            elif neg_syms:
                cell_color = "rgba(200, 100, 100, 0.5)"  # Light red
                cell_text = f"¬{'/'.join(neg_syms)}"
            
            # Draw hexagon
            fig.add_trace(go.Scatter(
                x=list(hex_x) + [hex_x[0]],
                y=list(hex_y) + [hex_y[0]],
                mode='lines',
                fill='toself',
                fillcolor=cell_color,
                line=dict(color='black', width=2),
                showlegend=False,
                hoverinfo='text',
                hovertext=f"Position ({row+1}:{col+1})"
            ))
            
            # Add position label
            annotations.append(dict(
                x=x, y=y,
                text=f"{row+1}:{col+1}",
                showarrow=False,
                font=dict(size=10, color='black'),
                yshift=12
            ))
            
            # Add symbol constraint
            if cell_text:
                annotations.append(dict(
                    x=x, y=y,
                    text=cell_text,
                    showarrow=False,
                    font=dict(size=14, color='black', family='Arial Black'),
                    yshift=-5
                ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x"),
        plot_bgcolor='white',
        height=300 + board_size * 50,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    fig.update_layout(annotations=annotations)
    
    return fig, legend_items


def create_symbol_match_visualization(interpretation):
    '''
    Creates a bar chart showing how many hypervector bits match each symbol.
    '''
    symbols = ['X (Black)', 'O (White)', '. (Empty)']
    
    pos_counts = [interpretation['positive_counts'].get(i, 0) for i in range(3)]
    neg_counts = [interpretation['negative_counts'].get(i, 0) for i in range(3)]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Positive (Must Have)',
        x=symbols,
        y=pos_counts,
        marker_color='green'
    ))
    
    fig.add_trace(go.Bar(
        name='Negative (Must NOT Have)',
        x=symbols,
        y=neg_counts,
        marker_color='red'
    ))
    
    fig.update_layout(
        title='Symbol Match Counts (Hypervector Bits)',
        barmode='group',
        xaxis_title='Symbol',
        yaxis_title='Matching Bits',
        height=300
    )
    
    return fig

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
        return None, None, None, None, None, None

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
        number_of_state_bits = model_dict.get('number_of_state_bits', 8)  # Default to 8
        number_of_literals = model_dict.get('number_of_literals', 64)  # hypervector_size * 2
        hypervector_size = number_of_literals // 2
        
        '''
        Extract hypervectors for symbol interpretation.
        '''
        hypervectors = model_dict.get('hypervectors', None)
        
        '''
        Processing 'ta_state' to count literals per clause.
        Use the stored number of clauses and infer the width dynamically
        so mismatched configurations do not break reshaping.
        '''
        ta_state_flat = model_dict['ta_state']
        if ta_state_flat.size % num_clauses != 0:
            raise ValueError(
                f"ta_state size {ta_state_flat.size} not divisible by number_of_clauses={num_clauses}"
            )
        ta_state_width = ta_state_flat.size // num_clauses
        ta_state_reshaped = ta_state_flat.reshape((num_clauses, ta_state_width))
        
        # Count literals by properly unpacking the packed bit representation
        literal_counts = []
        for row in ta_state_reshaped:
            literals = get_clause_literals(row, number_of_state_bits)
            literal_counts.append(len(literals))
        literal_counts = np.array(literal_counts)

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
        return None, None, None, None, None, None
    except Exception as e:
        st.error(f"Error analyzing model structure (ta_state/clause_weights): {e}")
        return None, None, None, None, None, None

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
    return parameters, df_clauses, ta_state_reshaped, number_of_state_bits, hypervectors, hypervector_size

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
    params, df_clauses, ta_state_reshaped, number_of_state_bits, hypervectors, hypervector_size = load_model_data(selected_model_path)

    '''
    Display the main dashboard only if the model data
    was successfully loaded.
    '''
    if params is not None and df_clauses is not None and ta_state_reshaped is not None:

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
        Click on a row to see the clause's literals.
        '''
        st.subheader(f"Raw Data for Top {top_n} Clauses")
        st.info("💡 Click on a row to see the literals that make up the clause.")
        
        # Add clause index to track which clause is selected
        df_display = df_clauses_sorted.head(top_n).copy()
        df_display['clause_index'] = df_display['clause_id'].str.replace('Clause_', '').astype(int)
        
        # Display dataframe with selection enabled
        selection = st.dataframe(
            df_display[['clause_id', 'literal_count', 'relevance_score']],
            use_container_width=True,
            selection_mode="single-row",
            on_select="rerun"
        )
        
        '''
        Section 3: Clause Literal Details
        When a clause is selected, display its literals.
        '''
        if selection and selection.selection and selection.selection.rows:
            selected_row_idx = selection.selection.rows[0]
            selected_clause_row = df_display.iloc[selected_row_idx]
            selected_clause_idx = int(selected_clause_row['clause_index'])
            selected_clause_id = selected_clause_row['clause_id']
            
            st.subheader(f"📋 Literals for {selected_clause_id}")
            
            # Get the literals for the selected clause
            clause_ta_state = ta_state_reshaped[selected_clause_idx]
            active_literals = get_clause_literals(clause_ta_state, number_of_state_bits)
            
            if active_literals:
                # Create a dataframe for the literals
                df_literals = pd.DataFrame(active_literals, columns=['Literal Index', 'TA State Value'])
                
                col_info, col_data = st.columns([1, 3])
                
                with col_info:
                    st.metric("Active Literals", len(active_literals))
                    st.metric("State Bits", number_of_state_bits)
                    threshold = 2 ** (number_of_state_bits - 1)
                    st.metric("Inclusion Threshold", threshold)
                    st.metric("Hypervector Size", hypervector_size)
                
                with col_data:
                    st.dataframe(df_literals, use_container_width=True, height=300)
                
                # -------------------------------------------------------
                # HEX BOARD VISUALIZATION
                # -------------------------------------------------------
                st.subheader(f"🎮 Hex Game Interpretation for {selected_clause_id}")
                
                if hypervectors is not None:
                    # Try to infer board size from model or use selector
                    # Check if board size is in the filename
                    board_match = re.search(r"board_(\d+)", selected_model_name)
                    default_board_size = int(board_match.group(1)) if board_match else 3
                    
                    board_size = st.selectbox(
                        "Select Hex Board Size:",
                        options=[3, 4, 5, 6],
                        index=[3, 4, 5, 6].index(default_board_size) if default_board_size in [3, 4, 5, 6] else 0,
                        key=f"board_size_{selected_clause_id}"
                    )
                    
                    # Interpret the literals in terms of Hex symbols
                    interpretation = interpret_literals_for_hex(
                        active_literals, 
                        hypervectors, 
                        hypervector_size,
                        board_size
                    )
                    
                    # Display interpretation summary
                    st.info("""
                    **How to read this visualization:**
                    - 🟢 **Green cells**: The clause requires this symbol to be present
                    - 🔴 **Red cells**: The clause requires this symbol to be absent (NOT)
                    - 🟡 **Yellow cells**: Contradiction - both present AND absent requirements
                    - ⬜ **Gray cells**: No specific constraint
                    
                    **Symbols:** X = Black (Player 1), O = White (Player 2), . = Empty
                    """)
                    
                    col_hex, col_stats = st.columns([2, 1])
                    
                    with col_hex:
                        # Create and display Hex board visualization
                        fig_hex, legend_items = create_hex_board_visualization(
                            board_size, 
                            interpretation,
                            title=f"Clause Pattern on {board_size}x{board_size} Hex Board"
                        )
                        st.plotly_chart(fig_hex, use_container_width=True)
                        
                        # Show legend
                        if legend_items:
                            st.write("**Clause Requirements:**")
                            for item in legend_items:
                                st.write(f"  {item}")
                    
                    with col_stats:
                        # Show symbol match statistics
                        st.write("**Symbol Match Analysis:**")
                        fig_matches = create_symbol_match_visualization(interpretation)
                        st.plotly_chart(fig_matches, use_container_width=True)
                        
                        # Show raw counts
                        st.write("**Hypervector Bit Matches:**")
                        for sym_idx, sym_name in enumerate(['X (Black)', 'O (White)', '. (Empty)']):
                            pos = interpretation['positive_counts'].get(sym_idx, 0)
                            neg = interpretation['negative_counts'].get(sym_idx, 0)
                            st.write(f"  {sym_name}: +{pos} / -{neg}")
                    
                    # Show contradictions warning if any
                    if interpretation['contradictions']:
                        st.warning(f"""
                        ⚠️ **Contradictions Detected!**
                        
                        The following symbols have both positive AND negative literals active:
                        {', '.join([['X', 'O', '.'][i] for i in interpretation['contradictions']])}
                        
                        This might indicate:
                        - Complex pattern matching behavior
                        - The clause is matching multiple possible scenarios
                        - Possible overfitting or noise in the learned pattern
                        """)
                    
                    # Show detailed bit breakdown
                    with st.expander("🔍 Detailed Bit Analysis"):
                        st.write(f"**Positive Literal Bits (0 to {hypervector_size-1}):** Symbol should be PRESENT")
                        st.write(f"Active positive bits: {interpretation['positive_bits'][:20]}{'...' if len(interpretation['positive_bits']) > 20 else ''}")
                        
                        st.write(f"**Negative Literal Bits ({hypervector_size} to {hypervector_size*2-1}):** Symbol should be ABSENT")
                        st.write(f"Active negative bits: {interpretation['negative_bits'][:20]}{'...' if len(interpretation['negative_bits']) > 20 else ''}")
                        
                        st.write("**Hypervector Mapping:**")
                        for sym_idx, sym_name in enumerate(['X', 'O', '.']):
                            bits = hypervectors[sym_idx].tolist() if sym_idx < len(hypervectors) else []
                            st.write(f"  {sym_name}: bits {bits}")
                else:
                    st.warning("Hypervectors not found in model. Cannot visualize Hex interpretation.")
                
                # Show additional visualization - histogram of literal indices
                st.write("**Literal Index Distribution:**")
                fig_literals = px.histogram(
                    df_literals,
                    x='Literal Index',
                    nbins=min(50, len(active_literals)),
                    title=f"Distribution of Active Literals in {selected_clause_id}"
                )
                fig_literals.update_layout(bargap=0.1)
                st.plotly_chart(fig_literals, use_container_width=True)
            else:
                st.warning(f"No active literals found in {selected_clause_id}.")

    else:
        st.error(f"Dashboard could not be loaded for {selected_model_name}.")