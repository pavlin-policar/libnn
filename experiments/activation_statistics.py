import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from libnn.losses import CategoricalCrossEntropy
from libnn.modules.activations import Softmax, Tanh, ReLU
from libnn.modules.layers import Linear

X = np.random.randn(1000, 500)


loss = CategoricalCrossEntropy()
activation = Tanh
initialization = 'xavier'

linear1 = Linear(500, 500, init=initialization)
activation1 = activation()
linear2 = Linear(500, 500, init=initialization)
activation2 = activation()
linear3 = Linear(500, 500, init=initialization)
activation3 = activation()
linear4 = Linear(500, 500, init=initialization)
activation4 = activation()
linear5 = Linear(500, 500, init=initialization)
activation5 = activation()
linear6 = Linear(500, 500, init=initialization)
activation6 = activation()
linear7 = Linear(500, 500, init=initialization)
activation7 = activation()
linear8 = Linear(500, 500, init=initialization)
activation8 = activation()
linear9 = Linear(500, 500, init=initialization)
activation9 = activation()

linear_final = Linear(500, 3)
softmax = Softmax()


n_bins = 15

# Perform the forward pass
z1 = activation1(linear1(X))
z2 = activation2(linear2(z1))
z3 = activation3(linear3(z2))
z4 = activation4(linear4(z3))
z5 = activation5(linear5(z4))
z6 = activation6(linear6(z5))
z7 = activation7(linear7(z6))
z8 = activation8(linear8(z7))
z9 = activation9(linear9(z8))

z = softmax(linear_final(z9))

# Plot activations
fig, ax = plt.subplots(3, 3, sharex=True, tight_layout=True)
sns.distplot(z1.ravel(), bins=n_bins, ax=ax[0][0])
sns.distplot(z2.ravel(), bins=n_bins, ax=ax[0][1])
sns.distplot(z3.ravel(), bins=n_bins, ax=ax[0][2])
sns.distplot(z4.ravel(), bins=n_bins, ax=ax[1][0])
sns.distplot(z5.ravel(), bins=n_bins, ax=ax[1][1])
sns.distplot(z6.ravel(), bins=n_bins, ax=ax[1][2])
sns.distplot(z7.ravel(), bins=n_bins, ax=ax[2][0])
sns.distplot(z8.ravel(), bins=n_bins, ax=ax[2][1])
sns.distplot(z9.ravel(), bins=n_bins, ax=ax[2][2])
plt.show()
