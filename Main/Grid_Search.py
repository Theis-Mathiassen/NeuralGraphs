from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid
import torch
from Classes import GCN, BaseModel, EvaluationMetricsData, StoredModel
from Search_Model import search_model
from Train_Test import train, test
from tqdm import trange
from torch_geometric.loader import DataLoader

from sklearn.model_selection import train_test_split

# -----
# INPUT
#   train_loader, test_loader : dataloader
#   device : torch.device
#   param_grid : library(str->list) that contains 'dropout_rate', 'hidden_channels', 'learning_rate'
# -----

def find_best(model, params, eval_data, best_accuracy, best_f1, best_roc, best_pr):
    if(eval_data.test_accuracy > best_accuracy.evalutation_metric):
        best_accuracy.update(eval_data.test_accuracy, model, params)
    
    if(eval_data.f1 > best_f1.evalutation_metric):
        best_f1.update(eval_data.f1, model, params)
        
    if(eval_data.roc > best_roc.evalutation_metric):
        best_roc.update(eval_data.roc, model, params)
        
    if(eval_data.pr > best_pr.evalutation_metric):
        best_pr.update(eval_data.roc, model, params)

def grid_search (dataset, device, param_grid, datasplit):
    # csv_class = Csv('filepath')
    # csv_class.open()

    #StoredModel for each of the evalutationMetrics
    best_accuracy = StoredModel()
    best_f1 = StoredModel()
    best_roc = StoredModel()
    best_pr = StoredModel()

    # Allocate data for training and remainder for testing 
    train_dataset = dataset[:datasplit]
    test_dataset = dataset[datasplit:]

    # iterates over each item defined in param_grid
    for params in ParameterGrid(param_grid):
        
        # Call search model which creates and runs the model based on the params. It returns the testData and the model used
        test_data, gridModel = search_model(params, train_dataset, test_dataset, device)

        for data in test_data:
            eval_data = EvaluationMetricsData(data)
            # Do something funny where params changes epochs 
            # csv_class.add(eval_data, params, time)
            find_best(gridModel, params, eval_data, best_accuracy, best_f1, best_roc, best_pr)
    print("Best accuracy: {} - with parameters {}", format((best_accuracy.evalutation_metric, best_accuracy.params)))


    


        


