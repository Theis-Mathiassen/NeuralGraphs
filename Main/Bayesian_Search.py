from Classes import GCN, BaseModel, EvaluationMetricsData, StoredModel
from Search_Model import search_model

import time  # Used to measure time
from bayes_opt import BayesianOptimization, UtilityFunction #From pip install bayesian-optimization



DATASPLIT = 150

# -----
# INPUT
#   train_loader, test_loader : dataloader
#   device : torch.device
#   param_grid : library(str->list) that contains 'dropout_rate', 'hidden_channels', 'learning_rate'
# -----
def bayesian_search (dataset, device, param_grid):
    # Initialize variables to keep track of the best model and its accuracy
    # Testing for test accuracy, f1 score, auc-roc, and auc-pr
    
    #StoredModel for each of the evalutationMetrics
    best_accuracy = StoredModel()
    best_f1 = StoredModel()
    best_roc = StoredModel()
    best_pr = StoredModel()


    #Params for bayesian search should be a min or a max value, or a list of string keys
    """   param_grid = {        
        'dropout_rate': [0.25, 0.75],
        'hidden_channels': [32, 64],
        'learning_rate': [0.005, 0.1],
        'batch_size' : [1, 64],
        'epochs' : [150, 300],
        'amount_of_layers' : [3, 5],
        'optimizer' : ['adam', 'sgd'],
        'activation_function' : ['relu', 'sigmoid']
        'pooling_algorithm' : ['mean', 'sum', 'max']
    } """

    # Allocate data for training and remainder for testing 
    train_dataset = dataset[:DATASPLIT]
    test_dataset = dataset[DATASPLIT:]

    print(len(train_dataset))
    print(len(test_dataset))

    # define function which will be run for each iteraiton 
    def black_box_function(dropout_rate, hidden_channels, learning_rate, batch_size, epochs, amount_of_layers, optimizer, activation_function, pooling_algorithm):
        # 1. Start by appropriately loading in params
        params = {}
        params['dropout_rate'] = dropout_rate
        params['hidden_channels'] = round(hidden_channels)      #Needs to be whole number so rounds
        params['learning_rate'] = learning_rate
        params['batch_size'] = round(batch_size)
        params['epochs'] = round(epochs)
        params['amount_of_layers'] = round(amount_of_layers)
        # Selecting from string keys. To do this we consider which element index of the array it is nearest, and pick that key string
        params['optimizer'] = param_grid['optimizer'][round(optimizer)] 
        params['activation_function'] = param_grid['activation_function'][round(activation_function)] 
        params['pooling_algorithm'] = param_grid['pooling_algorithm'][round(pooling_algorithm)] 
        
        
        # 2. Run model
        test_data, gridModel = search_model(params, train_dataset, test_dataset, device)

        eval_data = EvaluationMetricsData(test_data)
        
        # 3. Return score (performance value)
        return eval_data.accuracy
        
    # Set a range to optimize for. Min and max values are chosen for each hyper parameter
    pbounds  = {        
        'dropout_rate': [min(param_grid['dropout_rate']), max(param_grid['dropout_rate'])],
        'hidden_channels': [min(param_grid['hidden_channels']), max(param_grid['hidden_channels'])],
        'learning_rate': [min(param_grid['learning_rate']), max(param_grid['learning_rate'])],
        'batch_size' : [min(param_grid['batch_size']), max(param_grid['batch_size'])],
        'epochs' : [min(param_grid['epochs']), max(param_grid['epochs'])],
        'amount_of_layers' : [min(param_grid['amount_of_layers']), max(param_grid['amount_of_layers'])],
        # For keys it picks 0 to their length - 1. Such that it can pick from its indexs
        'optimizer' : [0, len(param_grid['optimizer']) - 1],
        'activation_function' : [0, len(param_grid['activation_function']) - 1], 
        'pooling_algorithm' : [0, len(param_grid['pooling_algorithm']) - 1]
    } 
    
    start = time.time()
    
    gbm_bo = BayesianOptimization(black_box_function, pbounds, random_state=111)
    gbm_bo.maximize(init_points=20, n_iter=4) #TO do, figure out what these mean
    
    params_gbm = gbm_bo.max['params']
    
    #Update params so they fit
    params_gbm['hidden_channels'] = round(params_gbm['hidden_channels'])      #Needs to be whole number so rounds
    params_gbm['batch_size'] = round(params_gbm['batch_size'])
    params_gbm['epochs'] = round(params_gbm['epochs'])
    params_gbm['amount_of_layers'] = round(params_gbm['amount_of_layers'])
    # Selecting from string keys. To do this we consider which element index of the array it is nearest, and pick that key string
    params_gbm['optimizer'] = param_grid['optimizer'][round(params_gbm['optimizer'])] 
    params_gbm['activation_function'] = param_grid['activation_function'][round(params_gbm['activation_function'])] 
    params_gbm['pooling_algorithm'] = param_grid['pooling_algorithm'][round(params_gbm['pooling_algorithm'])] 
    
    
    print(params_gbm)
    print('It takes %s minutes' % ((time.time() - start)/60))
    
    


    #     # #Print out results and potentially best options
    #     # print("Parameter configuration results:\n Configuration: {}\n Accuracy: {}.\n F1: {}.\n AUC ROC: {}.\n AUC PR: {}.\n".format(
    #     #     params, test_data.test_accuracy, eval_data.f1, eval_data.roc, eval_data.pr))
        
    #     # #Update best options if model outperforms. The update function replaces storedValues
    #     # if(test_data.test_accuracy > best_accuracy.evalutation_metric):
    #     #     best_accuracy.update(test_data.test_accuracy, gridModel, params)
        
    #     # if(eval_data.f1 > best_f1.evalutation_metric):
    #     #     best_f1.update(eval_data.f1, gridModel, params)
            
    #     # if(eval_data.roc > best_roc.evalutation_metric):
    #     #     best_roc.update(eval_data.roc, gridModel, params)
            
    #     # if(eval_data.pr > best_pr.evalutation_metric):
    #     #     best_pr.update(eval_data.roc, gridModel, params)

    # print("Best accuracy: {}.\n Model used: {}.\n With parameter configuration: {}".format(best_accuracy.evalutation_metric, best_accuracy.model, best_accuracy.params))
    # print("Best F1: {}.\n Model used: {}.\n With parameter configuration: {}".format(best_f1.evalutation_metric, best_f1.model, best_f1.params))
    # print("Best AUC ROC: {}.\n Model used: {}.\n With parameter configuration: {}".format(best_roc.evalutation_metric, best_roc.model, best_roc.params))
    # print("Best AUC PR: {}.\n Model used: {}.\n With parameter configuration: {}".format(best_pr.evalutation_metric, best_pr.model, best_pr.params))
