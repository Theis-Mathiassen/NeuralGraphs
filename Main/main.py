# Constants
EPOCHS = 150
LEARNING_RATE = 0.01
MANUAL_SEED = 12345
HIDDEN_NODE_COUNT = 64
DATASPLIT = 150
BATCH_SIZE_TRAINING = 64
BATCH_SIZE_TESTING = 64


from heatmap import heatMap
import os
import torch_geometric
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import pandas as pd

from Classes import AllData, BaseModel, GCN
from Train_Test import train, test
from Grid_Search import grid_search
from Bayesian_Search import bayesian_search
from Plot_func import HeatMap, GridBayesHist, GridBayesianComparison,HyperParamSearchPlot
from ReadCSV import GetHeatData,GetHistData,GetParamData

from tqdm import trange

# importing MUTAG
dataset = TUDataset(root='dataset/Mutag', name='MUTAG')

data_details = {
    "num_node_features": dataset.num_node_features,
    "num_edge_features": dataset.num_edge_features,
    "num_classes": dataset.num_classes,
    "num_node_labels": dataset.num_node_labels,
    "num_edge_labels": dataset.num_edge_labels,
}

print(data_details)

#  checks if a GPU is available for use w/ pytorch
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
#device = torch.device('cpu')
print(device)

# Set seed: manual
torch.manual_seed(MANUAL_SEED)
dataset = dataset.shuffle()

#parameter grid - set of hyper parameters and values for grid_search to iterate over
param_grid = {
    'dropout_rate': [0.25, 0.50, 0.75],
    'hidden_channels': [128],
    'learning_rate': [0.01],
    'batch_size' : [16, 32, 64, 150],
    'epochs' : [10, 50, 100, 200],
    'amount_of_layers' : [1, 2, 3, 9],
    'optimizer' : ['SGD', 'adam', 'rmsprop'],        #String key   'SGD', 'adam', 'RMSprop'
    'activation_function' : ['relu', 'sigmoid', 'tanh'], #'Relu', 'sigmoid', 'tanh'
    'pooling_algorithm' : ['mean', 'sum']  #'mean', 'sum', 'max'
}

heatMap()

#grid_search(dataset, device, param_grid, DATASPLIT, 'VestaNew_HC-128_LR-0.01')
startingPoints = 20
iterations = 20
bayesian_search(dataset, device, param_grid, startingPoints, iterations, read_logs=False, Seed=0)


#HeatMap(subsampled_results) # Plot clustermap 

#hyper_param = 'dropout_rate' # Hyper parameter to plot : 
#grid_data_p, grid_data_s, bayes_data_p, bayes_data_s = GetParamData(hyper_param,'roc', 75) # Get data
#GridBayesianComparison(grid_data_p, bayes_data_p, grid_data_s, bayes_data_s, hyper_param) # Plot data

#GridBayesHist()


