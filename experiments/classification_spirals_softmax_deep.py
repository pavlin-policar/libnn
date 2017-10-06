from experiments import plotting
from libnn import datasets
from libnn.losses import CategoricalCrossEntropy
from libnn.modules.activations import ReLU, Softmax
from libnn.modules.layers import Linear
from libnn.modules.module import Sequential
from libnn.modules.normalization import BatchNormalization

X, y = datasets.spirals(300)


hidden_layer_size = 10
initialization = 'random_normal'


loss = CategoricalCrossEntropy()

model = Sequential(
    Linear(2, hidden_layer_size, init=initialization),
    BatchNormalization(hidden_layer_size),
    ReLU(),
    Linear(hidden_layer_size, hidden_layer_size, init=initialization),
    BatchNormalization(hidden_layer_size),
    ReLU(),
    Linear(hidden_layer_size, hidden_layer_size, init=initialization),
    BatchNormalization(hidden_layer_size),
    ReLU(),
    Linear(hidden_layer_size, hidden_layer_size, init=initialization),
    BatchNormalization(hidden_layer_size),
    ReLU(),
    Linear(hidden_layer_size, hidden_layer_size, init=initialization),
    BatchNormalization(hidden_layer_size),
    ReLU(),
    Linear(hidden_layer_size, hidden_layer_size, init=initialization),
    BatchNormalization(hidden_layer_size),
    ReLU(),
    Linear(hidden_layer_size, hidden_layer_size, init=initialization),
    BatchNormalization(hidden_layer_size),
    ReLU(),
    Linear(hidden_layer_size, hidden_layer_size, init=initialization),
    BatchNormalization(hidden_layer_size),
    ReLU(),
    Linear(hidden_layer_size, hidden_layer_size, init=initialization),
    BatchNormalization(hidden_layer_size),
    ReLU(),
    Linear(hidden_layer_size, hidden_layer_size, init=initialization),
    BatchNormalization(hidden_layer_size),
    ReLU(),
    Linear(hidden_layer_size, 3, init=initialization),
    Softmax(),
)


learning_rate = 1
for epoch in range(500):
    # Perform the forward pass
    z = model(X)

    # Compute the error
    error = loss(z, y)
    print('Epoch %4d: %.4f' % (epoch, error))

    # Backpropagate the gradient
    gradient = loss.gradient()
    gradient = model.backward(gradient)

    # Update the weights
    for parameter in model.trainable_parameters():
        parameter -= learning_rate * parameter.grad

plotting.decision_boundary(X, y, model)
