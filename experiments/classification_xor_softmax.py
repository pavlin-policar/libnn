from experiments import plotting
from libnn import datasets
from libnn.losses import CategoricalCrossEntropy
from libnn.modules.activations import ReLU, Softmax
from libnn.modules.layers import Linear
from libnn.modules.module import Sequential

X, y = datasets.spirals(100)


loss = CategoricalCrossEntropy()

model = Sequential(
    Linear(2, 200),
    ReLU(),
    Linear(200, 3),
    Softmax(),
)


learning_rate = 1
for epoch in range(5000):
    # Perform the forward pass
    z = model(X)

    # Compute the error
    error = loss(z, y)
    print('Epoch %2d: %.4f' % (epoch, error))

    # Backpropagate the gradient
    gradient = loss.gradient()
    gradient = model.backward(gradient)

    # Update the weights
    for parameter in model.trainable_parameters():
        parameter -= learning_rate * parameter.grad

plotting.decision_boundary(X, y, model)
