import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def import_data():
    # remember to first run bash script: "sort spambase.data -R -0 spambase.data" in data directory
    path = "../spambase/spambase.data"
    data = pd.read_csv(path, header=None)
    return data


def normalize(data):
    return (data - data.mean(axis=0)) / data.std(axis=0)


def shuffle_index(data):
    indices = [i for i in range(data.shape[0])]
    random.shuffle(indices)
    return indices


def get_batch(data, indices, size, i):
    y = data.iloc[indices[i * size:(i + 1) * size], 57]
    X = data.iloc[indices[i * size:(i + 1) * size], 0:57]
    return y, X


def pred(X, params):
    return np.dot(X, params).ravel()


def err(y, pred):
    return np.subtract(y.ravel(), pred).reshape(y.shape[0], 1)


def lr_mse_loss(y, X, params):
    pred = np.dot(X, params).ravel()
    err = np.subtract(y.ravel(), pred).reshape(X.shape[0], 1)
    errsqr = np.sum(err ** 2)
    mse = errsqr / (2 * X.shape[0])
    return mse


def lr_gradient_loss(y, X, params):
    xtrans = np.transpose(X.values)
    predict = pred(X, params)
    error = err(y, predict)
    grad = np.dot(xtrans, error)
    return -2 * grad / X.shape[0]


def log_loss(y, X, params):
    y_hat = (1 / (1 + np.exp(-np.dot(X, params))))
    partA = np.dot(y, np.log(y_hat))
    ones = np.ones(shape=(X.shape[0], 1), dtype=float)
    partB = np.dot(ones.ravel() - y.values, np.log(ones.ravel() - y_hat.ravel()))
    return -1 / len(y) * (partA + partB)


config = {}
config['loss'] = lr_mse_loss
config['gradient_loss'] = lr_gradient_loss
config['alpha'] = 0.001
config['max_epochs'] = 10
config['convergence'] = 0.001
config['batch_size'] = 1


def sgd(data, c):
    epoch = 0
    fall_in_loss = 1000.0  # this is set to arbitrary high value so that the loop condition is satisfied at first run
    loss_over_epochs = []
    params_over_epochs = []
    params = np.zeros(shape=(57, 1), dtype=float)
    N = data.shape[0]
    n_of_batches = int(N / c['batch_size'])
    while (epoch < c['max_epochs']) and (fall_in_loss > c['convergence']):
        indices = shuffle_index(data)
        # calculate loss on full data:
        y, X = get_batch(data, indices, N, 0)
        loss_epoch_start = c['loss'](y, X, params)
        for n in range(n_of_batches):
            # obtain a batch at the nth position
            y, X = get_batch(data, indices, c['batch_size'], n)
            if n % 1000 == 0:
                print("Iter: " + str(n))
            params = params - c['alpha'] * c['gradient_loss'](y, X, params)

        # calculate loss on full data again
        y, X = get_batch(data, indices, N, 0)
        loss_epoch_end = c['loss'](y, X, params)
        print("Loss at end of an epoch: "+str(loss_epoch_end))
        # print(gradient_loss(y, X, params))
        loss_over_epochs.append(loss_epoch_end)
        params_over_epochs.append(params)
        fall_in_loss = np.linalg.norm(loss_epoch_start) - np.linalg.norm(loss_epoch_end)
        epoch += 1
    return loss_over_epochs, params_over_epochs


def custom_idx_ranges(data):
    dict = {}
    for i in range(0, 10):
        dict[i] = range(i, len(data), 10)
    return dict


def split_data(data, subsample_indices, excluded_subsample):
    test_sample_indices = subsample_indices[excluded_subsample]
    return data.drop(labels=test_sample_indices, axis=0), data.iloc[test_sample_indices]


def cv(raw_data, c):
    index_ranges = custom_idx_ranges(raw_data)
    train_losses = []
    test_losses = []
    df = pd.DataFrame()
    for i in range(0, len(index_ranges)):
        data_train, data_test = split_data(raw_data, index_ranges, i)
        losses, params = sgd(data_train, c=c)
        train_loss_series = pd.Series(losses)
        df = pd.concat([df, train_loss_series], ignore_index=True, axis=1)
        test_loss = c['loss'](data_test.iloc[:, 57], data_test.iloc[:, 0:57], params[-1])
        best_train_loss = losses[-1]
        train_losses.append(best_train_loss)
        test_losses.append(test_loss)

    mean_train_loss_over_iterations = df.mean(axis=1)
    return train_losses, test_losses, mean_train_loss_over_iterations


def draw_losses(dict_of_losses):
    for key, loss in dict_of_losses.iteritems():
        plt.plot(range(len(loss)), loss.values, label=("loss at alpha: " + key))
    plt.xlabel("Folds")
    plt.ylabel("Error")
    plt.legend()
    plt.show()


def draw_roc(fpr, tpr):
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("AUC: " + str(auc(fpr, tpr, reorder=True)))
    plt.show()


config = {}
config['loss'] = lr_mse_loss
config['gradient_loss'] = lr_gradient_loss
config['alpha'] = 0.001
config['max_epochs'] = 10
config['convergence'] = 0.001
config['batch_size'] = 20


def collect_cv_results(raw_data, c=config):
    index_ranges = custom_idx_ranges(raw_data)
    df_fpr = pd.DataFrame()
    df_tpr = pd.DataFrame()
    for i in range(0, len(index_ranges)):
        data_train, data_test = split_data(raw_data, index_ranges, i)
        losses, params = sgd(data_train, c)
        sh_index = shuffle_index(data_test)
        y, X = get_batch(data_test, sh_index, data_test.shape[0], 0)
        yhat = pred(X, params[-1])
        fpr, tpr, threshold = roc_curve(y, yhat)
        fpr = pd.DataFrame(fpr)
        tpr = pd.DataFrame(tpr)
        df_fpr = pd.concat([df_fpr, fpr], ignore_index=True, axis=1)
        df_tpr = pd.concat([df_tpr, tpr], ignore_index=True, axis=1)

    mean_fpr = df_fpr.mean(axis=1)
    mean_tpr = df_tpr.mean(axis=1)

    return mean_fpr, mean_tpr
