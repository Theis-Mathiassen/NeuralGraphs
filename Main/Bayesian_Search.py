from Classes import GCN, BaseModel, EvaluationMetricsData, StoredModel, CSVWriter
from Search_Model import search_model

from bayes_opt import BayesianOptimization #From pip install bayesian-optimization

from bayes_opt.logger import JSONLogger

import os
import datetime
from bayes_opt.event import Events
from bayes_opt.logger import JSONLogger
from bayes_opt.util import load_logs


DATASPLIT = 150





# -----
# INPUT
#   train_loader, test_loader : dataloader
#   device : torch.device
#   param_grid : library(str->list) that contains hyperparameters
#   init_points : the amount of random points to set before algorithm begins
#   n_iter : iterations of the bayesian optimization algorithm 
# -----
def bayesian_search (dataset, device, param_grid, init_points, n_iter, read_logs, Seed = 1, Path = "./result"):
    # Allocate data for training and remainder for testing 
    train_dataset = dataset[:DATASPLIT]
    test_dataset = dataset[DATASPLIT:]
    """
    print(len(train_dataset))
    print(len(test_dataset))
    """
    SeedPath = Path + str(Seed)

    BayWriter = CSVWriter(SeedPath, Seed=True)
    BayWriter.CSVOpen(Seed=True)

    # adjusts parameters such that they fit the model
    def adjust_params (activation_function, amount_of_layers, batch_size, dropout_rate, epochs, hidden_channels, learning_rate, optimizer, pooling_algorithm):
        updated_params = {}
        updated_params['dropout_rate'] = dropout_rate
        updated_params['hidden_channels'] = round(hidden_channels)      #Needs to be whole number so rounds
        updated_params['learning_rate'] = learning_rate
        updated_params['batch_size'] = round(batch_size)
        updated_params['epochs'] = round(epochs)
        updated_params['amount_of_layers'] = round(amount_of_layers)
        # Selecting from string keys. To do this we consider which element index of the array it is nearest, and pick that key string
        updated_params['optimizer'] = param_grid['optimizer'][round(optimizer)] 
        updated_params['activation_function'] = param_grid['activation_function'][round(activation_function)] 
        updated_params['pooling_algorithm'] = param_grid['pooling_algorithm'][round(pooling_algorithm)] 
        
        return updated_params

    # define function which will be run for each iteraiton 
    def black_box_function(dropout_rate, hidden_channels, learning_rate, batch_size, epochs, amount_of_layers, optimizer, activation_function, pooling_algorithm):
        # 1. Start by appropriately loading in params
        params = adjust_params(activation_function, amount_of_layers, batch_size, dropout_rate, epochs, hidden_channels, learning_rate, optimizer, pooling_algorithm)
        start = datetime.datetime.now()
        # 2. Run model
        test_data, gridModel = search_model(params, train_dataset, test_dataset, device)
        end = datetime.datetime.now()
        eval_data = EvaluationMetricsData(test_data)
        
        BayWriter.CSVWriteRow(params=params, eval=eval_data, time=end-start)
        
        # 3. Return score (performance value: set to AUC ROC)
        return eval_data.roc
        
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
    
    

    # Create Bayesian model
    bayesian_model = BayesianOptimization(black_box_function, pbounds, random_state=Seed)
    

    # Maximize for target
    # init_points means the amount of initial states which randomly will be chosen within the search space
    # n_iter refers to the amount of iterations bayesian optimization will do, that are searched in the most likely area to be good



    

    """if (read_logs and os.path.isfile(SeedPath)):
        # New optimizer is loaded with previously seen points
        load_logs(bayesian_model, logs=[SeedPath])
        logger = JSONLogger(path=SeedPath, reset=False)
        init_points = 0
    else:
        logger = JSONLogger(path=SeedPath)"""
    
    #bayesian_model.subscribe(Events.OPTIMIZATION_STEP, logger)

    # Results will be saved in ./logs.log
    bayesian_model.maximize(init_points, n_iter)
    
    #Find parameters for optimal model
    optimal_params = bayesian_model.max['params']

    #Update params so they fit their meanings 
    optimal_params = adjust_params(*optimal_params.values())
    
    #Print out best model performance (target) + params
    print('Target measure: '+str(bayesian_model.max['target']))
    print(optimal_params)

    BayWriter.CSVClose()


