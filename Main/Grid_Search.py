from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid
import torch
from Classes import GCN, BaseModel
from Train_Test import train, test
from tqdm import trange

from sklearn.model_selection import train_test_split


# constants
EPOCHS = 300


# -----
# INPUT
#   train_loader, test_loader : dataloader
#   device : torch.device
# -----
def grid_search (train_loader, test_loader, device):
    # Initialize variables to keep track of the best model and its accuracy
    best_model = None
    best_epoch_accuracy = 0.0
    best_params = None

    """   param_grid = {
        'dropout_rate': [0.25, 0.5, 0.75],
        'hidden_channels': [32, 64],
        'learning_rate': [0.1, 0.01, 0.005],
    } """

    # parameter grid - set of hyper parameters and corresponding ranges or values
    param_grid = {
        'dropout_rate': [0.25, 0.75],
        'hidden_channels': [ 32, 64],
        'learning_rate': [0.01]
    }

    # iterates over each item defined in param_grid
    for params in ParameterGrid(param_grid):

        gridModel = GCN(hidden_channels=params["hidden_channels"], dropout_rate=params["dropout_rate"], learning_rate=params["learning_rate"])
        gridModel.to(device)
        optimizer = torch.optim.Adam(gridModel.parameters(), lr=gridModel.learning_rate)
        loss_function = torch.nn.CrossEntropyLoss()
        baseModel = BaseModel(gridModel, loss_function, optimizer)


        for epoch in trange(0, EPOCHS):
            # TRAIN
            train_data = train(baseModel, train_loader, device)
            
        test_data = test(baseModel, test_loader, device)
            
        # Calculate average loss and accuracy for the entire epoch
        #avg_epoch_loss = epoch_loss / len(test_loader)
        #avg_epoch_accuracy = epoch_accuracy / len(test_loader)
        #test_accuracies_grid.append(avg_epoch_accuracy)


        print("Parameter configuration results:\n Configuration: {}\n Accuracy: {}.\n".format(params, test_data.test_accuracy))
        if test_data.test_accuracy > best_epoch_accuracy:
            best_epoch_accuracy = test_data.test_accuracy
            best_model = gridModel
            best_params = params

    print("Best accuracy: {}.\n Model used: {}.\n With parameter configuration: {}".format(best_epoch_accuracy, best_model, best_params))

