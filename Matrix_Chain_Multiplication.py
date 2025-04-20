import streamlit as st
import pandas as pd
import numpy as np

st.title("Matrix Chain Multiplication - Dynamic Programming")

# Input for matrix dimensions
input_str = st.text_input("Enter matrix dimensions (e.g., 5 6 4 2 3):", "5 6 4 2 3")

# Convert input to list of integers
try:
    p = list(map(int, input_str.strip().split()))
    n = len(p) - 1
except:
    st.error("Please enter valid integer dimensions")
    st.stop()

# Validate input
if len(p) < 2:
    st.warning("Enter at least two dimensions (e.g., 5 6 for one matrix).")
    st.stop()

# Display matrix dimensions
st.subheader("Matrix Dimensions")
matrices = [f"A{i+1}: {p[i]}×{p[i+1]}" for i in range(n)]
st.write(", ".join(matrices))

def matrix_chain_order(p):
    n = len(p) - 1
    m = [[0] * n for _ in range(n)]
    s = [[0] * n for _ in range(n)]

    for chain_len in range(2, n + 1):
        for i in range(n - chain_len + 1):
            j = i + chain_len - 1
            m[i][j] = float('inf')
            for k in range(i+1, j):
                cost = m[i][k] + m[k + 1][j] + p[i] * p[k + 1] * p[j + 1]
                if cost < m[i][j]:
                    m[i][j] = cost
                    s[i][j] = k
    return m, s

def generate_detailed_steps(p):
    n = len(p) - 1
    # Initialize memoization tables
    m = [[0] * n for _ in range(n)]
    s = [[0] * n for _ in range(n)]
    
    all_steps = []
    
    # Fill diagonal (base cases - single matrices)
    for i in range(n):
        m[i][i] = 0
        all_steps.append({
            "Step": len(all_steps) + 1,
            "Chain": f"A{i+1}",
            "i": i + 1,
            "j": i + 1,
            "Cost": 0,
            "Calculation": "Base case (single matrix)"
        })
    
    # Fill the table
    for chain_len in range(2, n + 1):
        for i in range(n - chain_len + 1):
            j = i + chain_len - 1
            m[i][j] = float('inf')
            
            chain_notation = f"A{i+1}...A{j+1}"
            calculations = []
            
            for k in range(i, j):
                left_chain = f"A{i+1}...A{k+1}" if i != k else f"A{i+1}"
                right_chain = f"A{k+2}...A{j+1}" if k+1 != j else f"A{j+1}"
                
                cost = m[i][k] + m[k+1][j] + p[i] * p[k+1] * p[j+1]
                calculations.append({
                    "k": k + 1,
                    "Split": f"({left_chain}) × ({right_chain})",
                    "Left Cost": m[i][k],
                    "Right Cost": m[k+1][j],
                    "Mult Cost": f"{p[i]} × {p[k+1]} × {p[j+1]} = {p[i] * p[k+1] * p[j+1]}",
                    "Total": cost,
                    "Is Best": cost < m[i][j] or m[i][j] == float('inf')
                })
                
                if cost < m[i][j]:
                    m[i][j] = cost
                    s[i][j] = k
            
            best_k = s[i][j] + 1
            all_steps.append({
                "Step": len(all_steps) + 1,
                "Chain": chain_notation,
                "i": i + 1,
                "j": j + 1,
                "Cost": m[i][j],
                "Best k": best_k,
                "Detailed Calculations": calculations
            })
    
    return m, s, all_steps

def print_optimal_parens(s, i, j):
    if i == j:
        return f"A{i+1}"
    else:
        return f"({print_optimal_parens(s, i, s[i][j])} × {print_optimal_parens(s, s[i][j]+1, j)})"

# Compute solutions
m, s, detailed_steps = generate_detailed_steps(p)

# Display DP cost table
st.subheader("Minimum Multiplication Cost Table (m[i,j])")
cost_df = pd.DataFrame(np.array(m))
cost_df.index = [f"A{i+1}" for i in range(n)]
cost_df.columns = [f"A{j+1}" for j in range(n)]
st.dataframe(cost_df)

# Display split position table
st.subheader("Optimal Split Position Table (s[i,j])")
split_df = pd.DataFrame(np.array(s))
split_df.index = [f"A{i+1}" for i in range(n)]
split_df.columns = [f"A{j+1}" for j in range(n)]
st.dataframe(split_df)

# Show final result
st.subheader("Optimal Parenthesization")
optimal_parens = print_optimal_parens(s, 0, n-1)
st.write(optimal_parens)
st.write(f"Minimum number of scalar multiplications: {m[0][n-1]}")

# Visualize steps
st.subheader("Detailed Steps")
show_details = st.checkbox("Show detailed calculations for each step", value=False)

for step in detailed_steps:
    with st.expander(f"Step {step['Step']}: Chain {step['Chain']} (m[{step['i']},{step['j']}] = {step['Cost']})"):
        if 'Calculation' in step:
            st.write(step['Calculation'])
        elif show_details and 'Detailed Calculations' in step:
            calcs = step['Detailed Calculations']
            for calc in calcs:
                status = "✅ BEST" if calc['Is Best'] else "❌"
                st.write(f"{status} k={calc['k']}: {calc['Split']}")
                st.write(f"   Left subtable cost: {calc['Left Cost']}")
                st.write(f"   Right subtable cost: {calc['Right Cost']}")
                st.write(f"   Multiplication cost: {calc['Mult Cost']}")
                st.write(f"   Total cost: {calc['Total']}")
                st.write("---")
