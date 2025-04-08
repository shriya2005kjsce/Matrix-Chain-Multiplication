import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import time
from IPython.display import clear_output, display as ipy_display

def matrix_chain_multiplication_visual(dimensions, delay=1):
    n = len(dimensions) - 1
    m = [[0 for _ in range(n)] for _ in range(n)]
    s = [[0 for _ in range(n)] for _ in range(n)]

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Matrix Chain Multiplication DP Table Filling", fontsize=16)

    st_fig = st.pyplot(fig) # Create a placeholder for the Matplotlib figure in Streamlit

    def draw_tables():
        axs[0].clear()
        axs[1].clear()
        axs[0].set_title("Cost Table (m)")
        axs[1].set_title("Split Table (s)")
        axs[0].imshow(m, cmap='Blues', vmin=0)
        axs[1].imshow(s, cmap='Oranges', vmin=0)

        for i in range(n):
            for j in range(n):
                if j >= i:
                    axs[0].text(j, i, f"{m[i][j] if m[i][j] != float('inf') else '∞'}", ha='center', va='center', fontsize=10)
                    axs[1].text(j, i, f"{s[i][j]}", ha='center', va='center', fontsize=10)

        axs[0].set_xticks(np.arange(n))
        axs[0].set_yticks(np.arange(n))
        axs[1].set_xticks(np.arange(n))
        axs[1].set_yticks(np.arange(n))
        st_fig.pyplot(fig) # Update the Streamlit placeholder with the new figure
        time.sleep(delay)

    # Fill DP tables
    for l in range(1, n):
        for i in range(n - l):
            j = i + l
            m[i][j] = float('inf')
            for k in range(i, j):
                cost = m[i][k] + m[k+1][j] + dimensions[i] * dimensions[k+1] * dimensions[j+1]
                if cost < m[i][j]:
                    m[i][j] = cost
                    s[i][j] = k
            draw_tables()

    plt.close(fig)
    return m, s

def print_optimal_parenthesization(s, i, j):
    if i == j:
        return f"A{i+1}"
    else:
        return f"({print_optimal_parenthesization(s, i, s[i][j])} × {print_optimal_parenthesization(s, s[i][j]+1, j)})"

def main():
    st.title("Matrix Chain Multiplication Visualizer")

    dimensions_input = st.text_input("Enter matrix dimensions (e.g., 5,4,6,2,7):", "5,4,6,2,7")
    delay = st.slider("Visualization Delay (seconds):", 0.1, 2.0, 1.0, 0.1)

    if st.button("Visualize"):
        try:
            dimensions = [int(d.strip()) for d in dimensions_input.split(',')]
            if len(dimensions) < 2:
                st.error("Please enter at least two dimensions.")
            else:
                m, s = matrix_chain_multiplication_visual(dimensions, delay=delay)
                n_matrices = len(dimensions) - 1
                st.write("\nMinimum number of scalar multiplications:", m[0][n_matrices - 1])
                st.write("Optimal parenthesization:", print_optimal_parenthesization(s, 0, n_matrices - 1))
        except ValueError:
            st.error("Invalid input. Please enter comma-separated integers for dimensions.")

if __name__ == "__main__":
    main()
