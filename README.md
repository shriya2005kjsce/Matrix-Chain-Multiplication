# 🧮 Matrix Chain Multiplication Visualizer

This little app helps you understand how the Matrix Chain Multiplication algorithm works — step by step. It's built with [Streamlit](https://streamlit.io/) and is designed to walk you through the dynamic programming process in a simple, interactive way.

## 🤔 What's This About?

If you've ever studied dynamic programming, you've probably come across this classic problem:

> Given a sequence of matrices, what's the most efficient way to multiply them?

Since matrix multiplication is associative, the order of multiplication affects the total number of operations. The goal is to **minimize the number of scalar multiplications**.

This app helps you:
- Input any set of matrix dimensions
- Watch the dynamic programming table fill up
- Understand exactly how and why each choice is made
- See the optimal multiplication order (with parentheses!)

## ✨ What You Can Do

- 📝 *Enter your own matrix dimensions* (e.g., 5, 4, 6, 2, 7)
- ⏭ *Step through* each phase of the algorithm one step at a time
- 👀 *Visualize* the cost and split tables in real time
- 🔍 *Understand* the math behind each decision
- 🧾 *Get the final result* with the best way to parenthesize the matrix chain

## 🚀 Try It Out
🌐 Link to open the deployable version of this repository: [https://shriyadhriti.streamlit.app/](https://shriyadhriti.streamlit.app/)
🧠 Script Reference: http://Matrix-Chain-Multiplication.py

## 🛠 How to Use It

### 🔧 Running It Locally
1. Clone this repo:
   ```bash
   git clone https://github.com/your-username/matrix-chain-streamlit.git
   cd matrix-chain-streamlit
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```
   
## 👥 Made By

- **Saniya Navelkar** – 16010123300  
- **Shreya Nair** – 16010123323  
- **Shriya Shetty** – 16010123327

