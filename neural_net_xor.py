import numpy as np

np.random.seed(0)

# Training data: XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)

# Network: 2 inputs -> 4 hidden neurons -> 1 output
W1 = np.random.randn(2, 4)
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1)
b2 = np.zeros((1, 1))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


lr = 1.0
for epoch in range(5001):
    # Forward pass
    h = sigmoid(X @ W1 + b1)
    out = sigmoid(h @ W2 + b2)
    loss = np.mean((out - y) ** 2)

    # Backward pass (backpropagation via the chain rule)
    d_out = 2 * (out - y) / len(X) * out * (1 - out)
    dW2 = h.T @ d_out
    db2 = d_out.sum(axis=0, keepdims=True)
    d_h = (d_out @ W2.T) * h * (1 - h)
    dW1 = X.T @ d_h
    db1 = d_h.sum(axis=0, keepdims=True)

    # Gradient descent update
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

    if epoch % 1000 == 0:
        print(f"epoch {epoch:5d}  loss {loss:.5f}")

print("\nPredictions:")
for inp, pred in zip(X, out):
    print(inp, "->", round(float(pred[0]), 3))
