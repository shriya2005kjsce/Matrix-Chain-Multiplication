import streamlit as st
import numpy as np
import pandas as pd

# Helper to print optimal parenthesis
def print_optimal_parenthesization(s, i, j):
    if i == j:
        return f"A{i+1}"
    else:
        return f"({print_optimal_parenthesization(s, i, s[i][j])} × {print_optimal_parenthesization(s, s[i][j]+1, j)})"

# Initialize session state
def initialize_session_state():
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'dimensions' not in st.session_state:
        st.session_state.dimensions = [5, 2, 4, 7]
    if 'm' not in st.session_state:
        st.session_state.m = None
    if 's' not in st.session_state:
        st.session_state.s = None
    if 'current_l' not in st.session_state:
        st.session_state.current_l = 1
    if 'current_i' not in st.session_state:
        st.session_state.current_i = 0
    if 'current_j' not in st.session_state:
        st.session_state.current_j = 1
    if 'current_k' not in st.session_state:
        st.session_state.current_k = 0
    if 'best_k' not in st.session_state:
        st.session_state.best_k = -1
    if 'best_cost' not in st.session_state:
        st.session_state.best_cost = float('inf')
    if 'step_phase' not in st.session_state:
        st.session_state.step_phase = 'start'
    if 'k_costs' not in st.session_state:
        st.session_state.k_costs = []
    if 'algorithm_complete' not in st.session_state:
        st.session_state.algorithm_complete = False
    if 'show_final_result' not in st.session_state:
        st.session_state.show_final_result = False
    if 'step_count' not in st.session_state:
        st.session_state.step_count = 0

# Reset algorithm for new input or restart
def reset_algorithm():
    st.session_state.initialized = True
    dimensions = st.session_state.dimensions
    n = len(dimensions) - 1
    
    st.session_state.m = [[0 for _ in range(n)] for _ in range(n)]
    st.session_state.s = [[0 for _ in range(n)] for _ in range(n)]
    
    st.session_state.current_l = 1
    st.session_state.current_i = 0
    st.session_state.current_j = 1
    st.session_state.current_k = 0
    st.session_state.best_k = -1
    st.session_state.best_cost = float('inf')
    st.session_state.step_phase = 'start'
    st.session_state.k_costs = []
    st.session_state.algorithm_complete = False
    st.session_state.show_final_result = False
    st.session_state.step_count = 0

# Full execution of the algorithm
def run_full_algorithm():
    dimensions = st.session_state.dimensions
    n = len(dimensions) - 1
    m = [[0 for _ in range(n)] for _ in range(n)]
    s = [[0 for _ in range(n)] for _ in range(n)]
    
    for l in range(1, n):
        for i in range(n - l):
            j = i + l
            m[i][j] = float('inf')
            for k in range(i, j):
                cost = m[i][k] + m[k+1][j] + dimensions[i] * dimensions[k+1] * dimensions[j+1]
                if cost < m[i][j]:
                    m[i][j] = cost
                    s[i][j] = k
    
    st.session_state.m = m
    st.session_state.s = s
    st.session_state.algorithm_complete = True
    st.session_state.show_final_result = True

# Step-by-step execution logic
def handle_next_step():
    if st.session_state.algorithm_complete:
        return

    st.session_state.step_count += 1

    n = len(st.session_state.dimensions) - 1
    
    if st.session_state.step_phase == 'start':
        i = st.session_state.current_i
        j = st.session_state.current_j
        st.session_state.m[i][j] = float('inf')
        st.session_state.current_k = i
        st.session_state.best_k = -1
        st.session_state.best_cost = float('inf')
        st.session_state.k_costs = []
        st.session_state.step_phase = 'evaluate_k'
    
    elif st.session_state.step_phase == 'evaluate_k':
        i = st.session_state.current_i
        j = st.session_state.current_j
        k = st.session_state.current_k
        dimensions = st.session_state.dimensions
        
        cost = (st.session_state.m[i][k] +
                st.session_state.m[k+1][j] +
                dimensions[i] * dimensions[k+1] * dimensions[j+1])
        
        st.session_state.k_costs.append((k, cost))
        
        if cost < st.session_state.best_cost:
            st.session_state.best_cost = cost
            st.session_state.best_k = k
        
        if k + 1 < j:
            st.session_state.current_k += 1
        else:
            st.session_state.step_phase = 'update_best'
    
    elif st.session_state.step_phase == 'update_best':
        i = st.session_state.current_i
        j = st.session_state.current_j
        st.session_state.m[i][j] = st.session_state.best_cost
        st.session_state.s[i][j] = st.session_state.best_k
        st.session_state.step_phase = 'next_cell'
    
    elif st.session_state.step_phase == 'next_cell':
        i = st.session_state.current_i
        j = st.session_state.current_j
        l = st.session_state.current_l
        
        if i + 1 < len(st.session_state.dimensions) - 1 - l:
            st.session_state.current_i += 1
            st.session_state.current_j += 1
            st.session_state.step_phase = 'start'
        else:
            st.session_state.current_l += 1
            if st.session_state.current_l < len(st.session_state.dimensions) - 1:
                st.session_state.current_i = 0
                st.session_state.current_j = st.session_state.current_l
                st.session_state.step_phase = 'start'
            else:
                st.session_state.algorithm_complete = True
                st.session_state.show_final_result = True
                st.session_state.step_phase = 'complete'

# Display m and s matrices
def display_matrix_tables():
    st.write("### 📊 Cost Matrix (m)")
    df_m = pd.DataFrame(st.session_state.m).fillna("")
    st.dataframe(df_m, use_container_width=True)

    st.write("### 🔁 Split Matrix (s)")
    df_s = pd.DataFrame(st.session_state.s).fillna("")
    st.dataframe(df_s, use_container_width=True)

# Final result output
def display_final_result():
    if st.session_state.show_final_result:
        st.write("## ✅ Optimal Parenthesization:")
        expr = print_optimal_parenthesization(st.session_state.s, 0, len(st.session_state.dimensions) - 2)
        st.success(expr)

        st.write("## 🧮 Minimum Multiplication Cost:")
        st.code(st.session_state.m[0][len(st.session_state.dimensions) - 2])

# MAIN APP
def main():
    st.set_page_config(page_title="Matrix Chain Multiplication Visualizer", layout="centered")
    st.title("📦 Matrix Chain Multiplication Visualizer")

    initialize_session_state()

    with st.expander("🛠️ Input Matrix Dimensions"):
        default = ", ".join(map(str, st.session_state.dimensions))
        user_input = st.text_input("Enter dimensions (comma-separated):", value=default)
        if st.button("Set Dimensions"):
            try:
                dims = list(map(int, user_input.split(",")))
                if len(dims) >= 2:
                    st.session_state.dimensions = dims
                    reset_algorithm()
                    st.success("Dimensions updated and algorithm reset.")
                else:
                    st.warning("Please enter at least two numbers.")
            except:
                st.error("Invalid input. Please enter integers separated by commas.")

    st.markdown("### 🔢 Step Number: **" + str(st.session_state.step_count) + "**")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("▶️ Next Step"):
            handle_next_step()
    with col2:
        if st.button("🚀 Run Full Algorithm"):
            run_full_algorithm()
    with col3:
        if st.button("🔄 Reset"):
            reset_algorithm()

    display_matrix_tables()
    display_final_result()

if __name__ == "__main__":
    main()
