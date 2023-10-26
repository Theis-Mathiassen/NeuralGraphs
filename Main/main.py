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


# Allocate data for training and remainder for testing 
train_dataset = dataset[:DATASPLIT]
test_dataset = dataset[DATASPLIT:]

# Define the data loaders. Used later for training, can be ignored for now
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE_TRAINING, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE_TESTING, shuffle=False)

print(len(train_dataset))
print(len(test_dataset))


grid_search(train_loader, test_loader, device)


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