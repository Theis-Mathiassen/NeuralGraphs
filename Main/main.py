from heatmap import heatMap
import torch
from Grid_Search import grid_search
from Bayesian_Search import bayesian_search
from torch_geometric.datasets import TUDataset
from ReadCSV import GetParamData
from Plot_func import GridBayesianComparison
#from PlotGridSearch import plot_auc_count 
from PlotGridSearch import plot_grid_search

# Constants
MANUAL_SEED = 12345
DATASPLIT = 150

# Constants for control flow
PERFORM_GRID_SEARCH = True
PERFORM_BAYESIAN = False
DISPLAY_GRID_SEARCH_HEATMAP = False
DISPLAY_GRID_SEARCH_GRAPHS= False
DISPLAY_HPO_COMPARISON = False

#DISPLAY_BAYES_HIST = False

# Import MUTAG dataset
dataset = TUDataset(root="dataset/Mutag", name="MUTAG")

data_details = {
    "num_node_features": dataset.num_node_features,
    "num_edge_features": dataset.num_edge_features,
    "num_classes": dataset.num_classes,
    "num_node_labels": dataset.num_node_labels,
    "num_edge_labels": dataset.num_edge_labels,
}

#  Checks if a CUDA GPU is available. If so, use it, otherwise default to CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Set seed to a constant and shuffle it
torch.manual_seed(MANUAL_SEED)
dataset = dataset.shuffle()

# This is where the hyperparameters, and hyperparameters values for the parameter grid are defined
param_grid = {
    'dropout_rate': [0.25, 0.50, 0.75],
    'hidden_channels': ["INSERT OWN"],
    'learning_rate': ["INSERT OWN"],
    'batch_size' : [16, 32, 64, 150],
    'epochs' : [10, 50, 100, 200],
    'amount_of_layers' : [1, 2, 3, 9],
    'optimizer' : ['SGD', 'adam', 'rmsprop'],        #String key   'SGD', 'adam', 'RMSprop'
    'activation_function' : ['relu', 'sigmoid', 'tanh'], #'Relu', 'sigmoid', 'tanh'
    'pooling_algorithm' : ['mean', 'sum']  #'mean', 'sum', 'max'
}

if PERFORM_GRID_SEARCH:
    grid_search(dataset, device, param_grid, DATASPLIT, "test")

if PERFORM_BAYESIAN:
    startingPoints = 20
    iterations = 20
    bayesian_search(dataset, device, param_grid, startingPoints, iterations, read_logs=False, Seed=0)


if DISPLAY_GRID_SEARCH_HEATMAP:
    heatMap()
    
if DISPLAY_GRID_SEARCH_GRAPHS:
    plot_grid_search()

if DISPLAY_HPO_COMPARISON:
    hyper_param = 'dropout_rate' # Hyper parameter to plot :
    grid_data_p, grid_data_s, bayes_data_p, bayes_data_s = GetParamData(hyper_param,'roc', 75) # Get data
    GridBayesianComparison(grid_data_p, bayes_data_p, grid_data_s, bayes_data_s, hyper_param) # Plot data

#if DISPLAY_BAYESIAN_HISTOGRAM:
#    GridBayesHist()
