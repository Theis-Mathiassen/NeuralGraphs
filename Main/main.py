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
    'dropout_rate': [0.25,0.75],
    'hidden_channels': [32],
    'learning_rate': [0.01, 0.1],
    'batch_size' : [16, 64],
    #'epochs' : [20, 50, 150],
    'amount_of_layers' : [2],
    'optimizer' : ['SGD', 'adam'],        #String key   'SGD', 'adam', 'RMSprop'
    'activation_function' : ['relu', 'sigmoid'], #'Relu', 'sigmoid', 'tanh'
    'pooling_algorithm' : ['mean', 'sum']  #'mean', 'sum', 'max'
}

#grid_search(dataset, device, param_grid)

startingPoints = 20;
iterations = 20
#bayesian_search(dataset, device, param_grid, startingPoints, iterations)

bigData = [0.587, 0.124, 0.893, 0.456, 0.789, 0.234, 0.901, 0.345, 0.678, 0.012,
0.543, 0.876, 0.321, 0.654, 0.987, 0.111, 0.222, 0.333, 0.444, 0.555,
0.666, 0.777, 0.888, 0.999, 0.135, 0.246, 0.357, 0.468, 0.579, 0.690,
0.801, 0.912, 0.025, 0.136, 0.247, 0.358, 0.469, 0.580, 0.691, 0.802,
0.913, 0.026, 0.137, 0.248, 0.359, 0.470, 0.581, 0.692, 0.803, 0.914,
0.027, 0.138, 0.249, 0.360, 0.471, 0.582, 0.693, 0.804, 0.915, 0.028,
0.139, 0.250, 0.361, 0.472, 0.583, 0.694, 0.805, 0.916, 0.029, 0.140,
0.251, 0.362, 0.473, 0.584, 0.695, 0.806, 0.917, 0.030, 0.141, 0.252,
0.363, 0.474, 0.585, 0.696, 0.807, 0.918, 0.031, 0.142, 0.253, 0.364,
0.475, 0.586, 0.697, 0.808, 0.919, 0.032, 0.143, 0.254, 0.365, 0.476]

HyperParamSearchPlot(bigData, "f1")

""" 
loss_f = torch.nn.CrossEntropyLoss()
hc_array = [5, 10, 20, 40, 64, 100] # Længde = 7
lr_array = [0.1, 0.01, 0.005, 0.003, 0.001] 
all_data_list = []


for i in range(0, len(lr_array)): # Laver 7 basis modeller der kan trænes og testes senere
    temp_all_data = AllData()
    gcn = GCN(in_features=dataset.num_node_features, hidden_channels=64) # Gemmer i array, da jeg er usikker på om python passer by reference eller value????
    gcn.to(device=device)
    base_model = BaseModel(gcn, loss_f, torch.optim.Adam(gcn.parameters(), lr=lr_array[i])) 
    for epoch in trange(0, EPOCHS):
        temp_all_data.insert_train_data(train(base_model, train_loader, device))
        temp_all_data.insert_test_data(test(base_model, test_loader, device))
    

    all_data_list.append(temp_all_data)
  
MultiPlotter(all_data_list, lr_array, "Learning Rate")

plt.show()
 """