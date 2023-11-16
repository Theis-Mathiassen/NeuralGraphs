# Constants
EPOCHS = 150
LEARNING_RATE = 0.01
MANUAL_SEED = 12345
HIDDEN_NODE_COUNT = 64
DATASPLIT = 150
BATCH_SIZE_TRAINING = 64
BATCH_SIZE_TESTING = 64


import os
import torch_geometric
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt


from Classes import AllData, BaseModel, GCN
from Train_Test import train, test
from Plot_func import MultiPlotter
from Grid_Search import grid_search
from Bayesian_Search import bayesian_search
from Plot_func import HyperParamSearchPlot

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
    'dropout_rate': [0.25],#,0.75],
    'hidden_channels': [32, 64],
    'learning_rate': [0.01],# 0.1],
    'batch_size' : [16],# 64],
    'epochs' : [50],#, 50, 150],
    'amount_of_layers' : [2],
    'optimizer' : ['SGD'],#, 'adam'],        #String key   'SGD', 'adam', 'RMSprop'
    'activation_function' : ['relu', 'sigmoid'], #'Relu', 'sigmoid', 'tanh'
    'pooling_algorithm' : ['mean', 'sum']  #'mean', 'sum', 'max'
}


grid_search(dataset, device, param_grid, DATASPLIT)

startingPoints = 20
iterations = 20
#bayesian_search(dataset, device, param_grid, startingPoints, iterations)