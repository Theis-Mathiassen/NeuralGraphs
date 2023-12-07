# Constants
EPOCHS = 150
LEARNING_RATE = 0.01
MANUAL_SEED = 12345
HIDDEN_NODE_COUNT = 64
DATASPLIT = 150
BATCH_SIZE_TRAINING = 64
BATCH_SIZE_TESTING = 64


import torch
from torch_geometric.datasets import TUDataset
from Bayesian_Search import bayesian_search

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

print(device)

# Set seed: manual
torch.manual_seed(MANUAL_SEED)
dataset = dataset.shuffle()

#parameter grid - set of hyper parameters and values for grid_search to iterate over
param_grid = {
    'dropout_rate': [0, 0.50, 0.99],
    'hidden_channels': [5, 32, 64, 128],
    'learning_rate': [1 ,0.01, 0.0001 ],
    'batch_size' : [32, 64, 150],
    'epochs' : [10, 50, 100, 200],
    'amount_of_layers' : [1, 2, 3, 15],
    'optimizer' : ['SGD', 'adam', 'rmsprop'],        #String key   'SGD', 'adam', 'RMSprop'
    'activation_function' : ['relu', 'sigmoid', 'tanh'], #'Relu', 'sigmoid', 'tanh'
    'pooling_algorithm' : ['mean', 'sum']  #'mean', 'sum', 'max'
}

startingPoints = 2
iterations = 2

for i in range(1):
    bayesian_search(dataset, device, param_grid, startingPoints, iterations, read_logs=False, Seed=i)


