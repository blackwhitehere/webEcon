import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt


# remember to run bash script: sort spambase.data -R -0 spambase.data


def import_data():
    path = "../spambase/spambase.data"
    data = pd.read_csv(path, header=None)
    return data


def normalize(data):
    return (data - data.mean(axis=0)) / data.std(axis=0)


def get_batch(data, size):
    indices = []
    for i in range(0, size):
        index = random.randint(0, len(data))
        indices.append(index)

    y = data.iloc[indices[0:size], 57]
    X = data.iloc[indices[0:size], 0:57]
    return y, X


def lr_gradient_loss(y, X, params):
    N = X.shape[0]
    sum = X.iloc[0, :]
    y=y.values
    for n in range(N):
        error = y[n]-np.dot(np.transpose(params), X.iloc[n, :])
        if n==1:
            sum = sum*error
        else:
            sum = sum + error*X.iloc[n, :].values
    return -1/N * sum.values


def lr_mse_loss(y, X, params):
    errors = (y.values - np.dot(X.values, params).ravel())
    return 1 / (2 * len(y)) * (np.dot(errors, errors))


def log_loss(y, X, params):
    y_hat = (1 / (1 + np.exp(-np.dot(X, params))))
    partA = np.dot(y, np.log(y_hat))
    ones = np.ones(shape=(X.shape[0], 1), dtype=float)
    partB = np.dot(ones.ravel() - y.values, np.log(ones.ravel() - y_hat.ravel()))
    return -1 / len(y) * (partA + partB)


def sgd(data, loss, gradient_loss, alpha, max_epoch, convergence, batch_size):
    epoch = 0
    fall_in_loss = 0.0
    loss_over_epochs = []
    params_over_epochs = []
    params = 0
    while (epoch < max_epoch) or (fall_in_loss < convergence):
        if params is not 0:
            params = np.zeros(shape=(data.shape[1], 1), dtype=float)
        y, X = get_batch(data, len(data))
        loss_epoch_start = loss(y, X, params)
        for i in range(0, len(data)):
            y, X = get_batch(data, batch_size)
            if i % 1000 == 0:
                print("Iter: " + str(i))
                print("Loss: " + str(loss_epoch_start))
            params = params + alpha * gradient_loss(y, X, params)
        loss_epoch_end = loss(y, X, params)
        loss_over_epochs.append(loss_epoch_end)
        params_over_epochs.append(params)
        fall_in_loss = loss_epoch_start - loss_epoch_end

    return loss_over_epochs, params_over_epochs


def custom_idx_ranges(data):
    dict = {}
    for i in range(0, 10):
        dict[i] = range(0, len(data), 10)
    return dict


def split_data(data, subsample_indices, excluded_subsample):
    test_sample_indices = subsample_indices[excluded_subsample]
    return data.drop(labels=test_sample_indices, axis=0), data.iloc[test_sample_indices]


def cv(raw_data, loss, gradient_loss, alpha, max_epochs=10, convergence=0.001, batch_size=1):
    index_ranges = custom_idx_ranges(raw_data)
    train_losses = []
    test_losses = []
    for i in range(0, len(index_ranges)):
        data_train, data_test = split_data(raw_data, index_ranges, i)
        losses, params = sgd(data_train, loss, gradient_loss, alpha, max_epochs, convergence, batch_size)
        test_loss = loss(data_test.iloc[:, 58], data_test.iloc[:, 0:57])
        best_train_loss = losses[-1]
        train_losses.append(best_train_loss)
        test_losses.append(test_loss)

    return train_losses, test_losses


def draw_tran_test_loss(train_losses, test_losses):
    plt.plot(range(0, len(train_losses)), train_losses, legend="training loss")
    plt.plot(range(0, len(train_losses)), test_losses, legend="test loss")
    plt.legend()
    plt.show()
