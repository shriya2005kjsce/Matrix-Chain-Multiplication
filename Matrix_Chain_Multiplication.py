import matplotlib.pyplot as plt
import numpy as np
import time
from IPython.display import clear_output, display

def matrix_chain_multiplication_visual(dimensions, delay=1):
    n = len(dimensions) - 1
    m = [[0 for _ in range(n)] for _ in range(n)]
    s = [[0 for _ in range(n)] for _ in range(n)]

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Matrix Chain Multiplication DP Table Filling", fontsize=16)

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
        display(fig)
        plt.pause(delay)
        clear_output(wait=True)

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

# Example run
dimensions = [5, 4, 6, 2, 7]  # A1(5x4), A2(4x6), A3(6x2), A4(2x7)
m, s = matrix_chain_multiplication_visual(dimensions, delay=1)

print("\nMinimum number of scalar multiplications:", m[0][len(dimensions)-2])
print("Optimal parenthesization:", print_optimal_parenthesization(s, 0, len(dimensions)-2))
