import streamlit as st
import pandas as pd

st.title("Matrix Chain Multiplication - Dynamic Programming")

# Input for matrix dimensions
input_str = st.text_input("Enter matrix dimensions (e.g., 5 6 4 2 3):", "5 6 4 2 3")

# Convert input to list of integers
p = list(map(int, input_str.strip().split()))
n = len(p) - 1

# Validate input
if len(p) < 2:
    st.warning("Enter at least two dimensions (e.g., 5 6 for one matrix).")
    st.stop()

@st.cache_data
def matrix_chain_order(p):
    n = len(p) - 1
    m = [[0] * n for _ in range(n)]
    s = [[0] * n for _ in range(n)]

    for chain_len in range(2, n + 1):
        for i in range(n - chain_len + 1):
            j = i + chain_len - 1
            m[i][j] = float('inf')
            for k in range(i, j):
                cost = m[i][k] + m[k + 1][j] + p[i] * p[k + 1] * p[j + 1]
                if cost < m[i][j]:
                    m[i][j] = cost
                    s[i][j] = k
    return m, s

@st.cache_data
def generate_steps(p):
    m, _ = matrix_chain_order(p)
    n = len(p) - 1
    steps = []
    for chain_len in range(2, n + 1):
        for i in range(n - chain_len + 1):
            j = i + chain_len - 1
            for k in range(i, j):
                cost = m[i][k] + m[k + 1][j] + p[i] * p[k + 1] * p[j + 1]
                steps.append({
                    "i": i + 1,
                    "j": j + 1,
                    "k": k + 1,
                    "Cost": cost
                })
    return steps

m, s = matrix_chain_order(p)
steps = generate_steps(p)

# Display matrix dimensions
st.subheader("Matrix Dimensions")
matrices = [f"A{i+1}: {p[i]}×{p[i+1]}" for i in range(n)]
st.write(", ".join(matrices))

# Display DP cost table
st.subheader("Minimum Multiplication Cost Table (m[i][j])")
df_cost = pd.DataFrame(m)
df_cost.index = [f"A{i+1}" for i in range(n)]
df_cost.columns = [f"A{j+1}" for j in range(n)]
st.dataframe(df_cost)

# Display optimal split table
st.subheader("Optimal Split Table (s[i][j])")
df_split = pd.DataFrame(s)
df_split.index = [f"A{i+1}" for i in range(n)]
df_split.columns = [f"A{j+1}" for j in range(n)]
st.dataframe(df_split)

# Show steps
st.subheader("Computation Steps (Recurrence Relation Applications)")
st.dataframe(pd.DataFrame(steps))
