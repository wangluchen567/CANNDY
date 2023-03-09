import numpy as np
import matplotlib.pyplot as plt

from MLP import MLP
from Core.Loss import MSELoss
from Core.Optimizer import Adam

if __name__ == '__main__':
    X = np.random.uniform(0, 10, size=(100, 1))
    Y = np.sin(X)
    model = MLP(1, 1, [8, 16, 8])

    optimizer = Adam(model=model, learning_rate=0.01)
    for epoch in range(5000):
        input = X.T
        truth = Y.T
        optimizer.zero_grad()
        output = model.forward(input)
        Loss = MSELoss(model, truth, output)
        mse_loss = Loss.forward()
        print(epoch + 1, mse_loss)
        Loss.backward()
        optimizer.step()

    X = np.arange(0, 10, 0.01).reshape(1, -1)
    Y = np.sin(X)
    output = model.forward(X)
    plt.figure()
    plt.plot(X.flatten(), Y.flatten(), c='red')
    plt.plot(X.flatten(), output.flatten(), c='blue')
    plt.show()
