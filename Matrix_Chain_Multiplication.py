import matplotlib.pyplot as plt
import numpy as np

def get_matrix_dimensions():
    print("Enter matrix dimensions as space-separated values (e.g., 5 4 6 2 7):")
    dims = list(map(int, input("Dimensions: ").strip().split()))
    if len(dims) < 2:
        raise ValueError("You need at least two dimensions (one matrix).")
    return dims

def print_optimal_parenthesization(s, i, j):
    if i == j:
        return f"A{i+1}"
    else:
        return f"({print_optimal_parenthesization(s, i, s[i][j])} × {print_optimal_parenthesization(s, s[i][j]+1, j)})"

def matrix_chain_multiplication_visual(dimensions):
    n = len(dimensions) - 1
    m = [[0 for _ in range(n)] for _ in range(n)]
    s = [[0 for _ in range(n)] for _ in range(n)]
    history = []  # Store step-by-step info

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Matrix Chain Multiplication Simulation", fontsize=18)

    steps = []  # All steps (i, j, k) to walk through
    for l in range(1, n):
        for i in range(n - l):
            j = i + l
            for k in range(i, j):
                steps.append((i, j, k))

    step_index = 0
    step_counter = 1

    while 0 <= step_index < len(steps):
        i, j, k = steps[step_index]
        cost = m[i][k] + m[k+1][j] + dimensions[i] * dimensions[k+1] * dimensions[j+1]

        # Save old value for rollback
        old_cost = m[i][j]
        old_split = s[i][j]

        if cost < m[i][j] or m[i][j] == 0:
            m[i][j] = cost
            s[i][j] = k

        # Add to history
        history.append((step_counter, f"(A{i+1} × ... × A{j+1}) | k = {k+1} | cost = {cost}"))

        # Draw tables
        axs[0].clear()
        axs[1].clear()
        axs[2].clear()

        axs[0].axis('off')
        axs[0].text(0.5, 0.8, f"Step {step_counter}", fontsize=18, ha='center')
        axs[0].text(0.5, 0.5, f"Evaluating:\n(A{i+1} × ... × A{j+1})", fontsize=14, ha='center')
        axs[0].text(0.5, 0.2, f"Split at k = {k+1}", fontsize=12, ha='center')

        axs[1].set_title("Cost Table (m)")
        axs[1].imshow(m, cmap='Blues', vmin=0)
        for x in range(n):
            for y in range(n):
                if y >= x:
                    val = m[x][y] if m[x][y] != float('inf') else '∞'
                    axs[1].text(y, x, str(val), ha='center', va='center', fontsize=10)
        axs[1].set_xticks(np.arange(n))
        axs[1].set_yticks(np.arange(n))

        axs[2].set_title("Split Table (s)")
        axs[2].imshow(s, cmap='Oranges', vmin=0)
        for x in range(n):
            for y in range(n):
                if y >= x:
                    axs[2].text(y, x, f"{s[x][y]}", ha='center', va='center', fontsize=10)
        axs[2].set_xticks(np.arange(n))
        axs[2].set_yticks(np.arange(n))

        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)

        # Print history log
        print("\n🔁 History so far:")
        for h in history[-5:]:  # last 5 steps
            print(f"  Step {h[0]}: {h[1]}")

        # User input for navigation
        user_input = input("\nPress Enter for next, or type 'b' for back: ").strip().lower()
        if user_input == 'b' and step_index > 0:
            # Undo last update
            m[i][j] = old_cost
            s[i][j] = old_split
            history.pop()
            step_index -= 1
            step_counter -= 1
        else:
            step_index += 1
            step_counter += 1

    plt.close(fig)
    return m, s

# Main driver
if __name__ == "__main__":
    dimensions = get_matrix_dimensions()
    m, s = matrix_chain_multiplication_visual(dimensions)

    print("\n✅ Simulation complete!")
    print("Minimum number of scalar multiplications:", m[0][len(dimensions)-2])
    print("Optimal parenthesization:", print_optimal_parenthesization(s, 0, len(dimensions)-2))
