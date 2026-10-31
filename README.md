# Tiny Neural Network (XOR)

A minimal neural network written from scratch in NumPy, with no ML framework.
It learns the XOR function, which a single neuron can't solve but a network
with one hidden layer can.

## How it works

- Architecture: 2 inputs -> 4 hidden neurons -> 1 output (sigmoid activations)
- Loss: mean squared error
- Training: backpropagation + gradient descent, written by hand

## Run it

```bash
pip install -r requirements.txt
python neural_net_xor.py
```

## Expected output

The loss drops from about 0.33 to under 0.001, and the predictions
approach 0, 1, 1, 0 for the four XOR inputs.
