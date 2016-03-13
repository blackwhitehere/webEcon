import utility as u
import numpy as np

def repeat4to7(data, config):
    #STEP 5:
    #default option is to use sgd with batch size = 1 and linear regression loss
    alphas = [0.001, 0.005, 0.0001]
    cv_mean_loss = {}
    dict_of_losses = {}
    #config=u.config
    for alpha in alphas:
        config['alpha']=alpha
        train_loss, test_loss, cv_losses_over_epochs = u.cv(raw_data=data, c=config)
        dict_of_losses[str(alpha)] = cv_losses_over_epochs
        avg_test_loss = np.mean(test_loss)
        cv_mean_loss[str(alpha)] = avg_test_loss

    print(cv_mean_loss)
    best_alpha = min(cv_mean_loss, key=cv_mean_loss.get)
    print("The best alpha is: " + str(best_alpha))
    u.draw_losses(dict_of_losses=dict_of_losses)

    #STEP 6:

    mean_fpr, mean_tpr = u.collect_cv_results(data, config)
    u.draw_roc(mean_fpr, mean_tpr)