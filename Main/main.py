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

bigData = {0.4, 0.34, 0.3, 0.4, 0.45, 0.5, 0.51, 0.52, 0.55, 0.56, 0.6, 0.7, 0.69, 0.73, 0.7, 0.79, 0.8, 0.75, 0.74, 0.76, 0.7, 0.69, 0.68, 0.69, 0.65}

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