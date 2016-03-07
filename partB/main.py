import utility as u
import numpy as np

data = u.import_data()
data.iloc[:, 0:57] = u.normalize(data.iloc[:, 0:57])

#y, X = u.get_batch(data, 2)
#print(y.values)
#print(X.head())


alpha = 0.01
max_epochs = 5
convergence = 0.1
batch_size = 1
train_losses, test_losses = u.cv(
    data, u.lr_mse_loss, u.lr_gradient_loss, alpha, max_epochs, convergence, batch_size)

u.draw_tran_test_loss(train_losses, test_losses)