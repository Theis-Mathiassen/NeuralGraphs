from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid
import torch
from Classes import GCN, BaseModel, EvaluationMetricsData, StoredModel
from Train_Test import train, test
from tqdm import trange
from torch_geometric.loader import DataLoader

from sklearn.model_selection import train_test_split

DATASPLIT = 150

# -----
# INPUT
#   train_loader, test_loader : dataloader
#   device : torch.device
#   param_grid : library(str->list) that contains 'dropout_rate', 'hidden_channels', 'learning_rate'
# -----
def grid_search (dataset, device, param_grid):
    # Initialize variables to keep track of the best model and its accuracy
    # Testing for test accuracy, f1 score, auc-roc, and auc-pr
    
    #StoredModel for each of the evalutationMetrics
    best_accuracy = StoredModel()
    best_f1 = StoredModel()
    best_roc = StoredModel()
    best_pr = StoredModel()
    
    best_model_accuracy = None
    best_epoch_accuracy = 0.0
    best_params_accuracy = None

    best_model_f1 = None
    best_epoch_f1 = 0.0
    best_params_f1 = None
    
    best_model_roc = None
    best_epoch_roc = 0.0
    best_params_roc = None
    
    best_model_pr = None
    best_epoch_pr = 0.0
    best_params_pr = None

    """   param_grid = {
        'dropout_rate': [0.25, 0.5, 0.75],
        'hidden_channels': [32, 64],
        'learning_rate': [0.1, 0.01, 0.005],
        'batch_size' : [1, 32, 64],
        'epochs' : [150, 300],
        'amount_of_layers' : [3,4,5],
        'optimizer' : ['adam', 'sgd'],
        'activation_function' : ['relu', 'sigmoid']
        'pooling_algorithm' : ['mean', 'sum', 'max']
    } """

    # Allocate data for training and remainder for testing 
    train_dataset = dataset[:DATASPLIT]
    test_dataset = dataset[DATASPLIT:]

    print(len(train_dataset))
    print(len(test_dataset))

    # iterates over each item defined in param_grid
    for params in ParameterGrid(param_grid):
        
        # Define the data loaders, this uses batch_size
        train_loader = DataLoader(dataset=train_dataset, batch_size=params["batch_size"], shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=params["batch_size"], shuffle=False)

        gridModel = GCN(hidden_channels=params["hidden_channels"], dropout_rate=params["dropout_rate"], learning_rate=params["learning_rate"], 
                        activation_function=params["activation_function"], amount_of_layers=params["amount_of_layers"], pooling_algorithm=params["pooling_algorithm"])
        gridModel.to(device)

        #Deciding which optimizer to use
        optimizer = None
        if(params["optimizer"].lower() == 'sgd'): optimizer = torch.optim.SGD(gridModel.parameters(), lr=gridModel.learning_rate)
        elif(params["optimizer"].lower() == 'adam'): optimizer = torch.optim.Adam(gridModel.parameters(), lr=gridModel.learning_rate)
        elif(params["optimizer"].lower() == 'rmsprop'): optimizer = torch.optim.RMSprop(gridModel.parameters(), lr=gridModel.learning_rate)
        else : raise Exception("Invalid optimizer name: " + str(params["optimizer"]))

        loss_function = torch.nn.CrossEntropyLoss()
        baseModel = BaseModel(gridModel, loss_function, optimizer)


        for epoch in trange(0, params["epochs"]):
            # TRAIN
            train_data = train(baseModel, train_loader, device)
            
        test_data = test(baseModel, test_loader, device)
            
        # Calculate average loss and accuracy for the entire epoch
        #avg_epoch_loss = epoch_loss / len(test_loader)
        #avg_epoch_accuracy = epoch_accuracy / len(test_loader)
        #test_accuracies_grid.append(avg_epoch_accuracy)
        
        #Get evaluation data from test_data
        eval_data = EvaluationMetricsData(test_data)

        #Print out results and potentially best options
        print("Parameter configuration results:\n Configuration: {}\n Accuracy: {}.\n F1: {}.\n AUC ROC: {}.\n AUC PR: {}.\n".format(
            params, test_data.test_accuracy, eval_data.f1, eval_data.roc, eval_data.pr))
        
        #Update best options if model outperforms. The update function replaces storedValues
        if(test_data.test_accuracy > best_accuracy.evalutation_metric):
            best_accuracy.update(test_data.test_accuracy, gridModel, params)
        
        if(eval_data.f1 > best_f1.evalutation_metric):
            best_f1.update(eval_data.f1, gridModel, params)
            
        if(eval_data.roc > best_roc.evalutation_metric):
            best_roc.update(eval_data.roc, gridModel, params)
            
        if(eval_data.pr > best_pr.evalutation_metric):
            best_pr.update(eval_data.roc, gridModel, params)

    print("Best accuracy: {}.\n Model used: {}.\n With parameter configuration: {}".format(best_accuracy.evalutation_metric, best_accuracy.model, best_accuracy.params))
    print("Best F1: {}.\n Model used: {}.\n With parameter configuration: {}".format(best_f1.evalutation_metric, best_f1.model, best_f1.params))
    print("Best AUC ROC: {}.\n Model used: {}.\n With parameter configuration: {}".format(best_roc.evalutation_metric, best_roc.model, best_roc.params))
    print("Best AUC PR: {}.\n Model used: {}.\n With parameter configuration: {}".format(best_pr.evalutation_metric, best_pr.model, best_pr.params))
