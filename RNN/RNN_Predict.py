import copy
import numpy as np
import matplotlib.pyplot as plt

from Core.Loss import MSELoss
from Core.Optimizer import Adam
from Core.Activation import Tanh
from Core.Module import RNNModel

if __name__ == '__main__':
    np.random.seed(0)
    num_samples, num_steps = 100, 10
    X = np.linspace(0, 10, num_samples * (num_steps + 1))
    Y = np.sin(X).reshape(num_samples, -1)
    data_x = Y[:, :-1].reshape(-1, num_steps, 1)
    data_y = Y[:, -1].reshape(-1, 1)
    model = RNNModel(input_size=1,
                     rnn_hidden_size=8,
                     num_layers=2,
                     linear_hidden_sizes=None,
                     output_size=1,
                     batch_first=True)

    optimizer = Adam(model=model, learning_rate=0.01)
    num_epochs = 1000
    for epoch in range(num_epochs):
        output = model.forward(data_x)
        Loss = MSELoss(model, data_y, output)
        mse_loss = Loss.forward()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {mse_loss.item()}')
        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()
