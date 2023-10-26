# Constants
EPOCHS = 300
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

from Classes import AllData, BaseModel, GCN
from Train_Test import train, test
from Plot_func import MultiPlotter

from tqdm import trange


dataset = TUDataset(root='dataset/Mutag', name='MUTAG')

data_details = {
    "num_node_features": dataset.num_node_features,
    "num_edge_features": dataset.num_edge_features,
    "num_classes": dataset.num_classes,
    "num_node_labels": dataset.num_node_labels,
    "num_edge_labels": dataset.num_edge_labels,
}


print(data_details)

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    # MPS is currently slower than CPU due to missing int64 min/max ops
    device = torch.device('cpu')
else:
    device = torch.device('cpu')
#device = torch.device('cpu')
print(device)



dataset = dataset.shuffle()


# Allocate for training
train_dataset = dataset[:DATASPLIT]
# Allocate the remainder for testing
test_dataset = dataset[DATASPLIT:]

# Define the data loaders. Used later for training, can be ignored for now
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE_TRAINING, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE_TESTING, shuffle=False)

print(len(train_dataset))
print(len(test_dataset))



#gcn = GCN(in_features=dataset.num_node_features, hidden_channels=HIDDEN_NODE_COUNT)
#gcn.to(device=device)
loss_f = torch.nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(gcn.parameters(), lr=LEARNING_RATE)

#model1 = BaseModel(gcn, loss_f, optimizer)

hc_array = [5, 10]# 20, 40, 64, 100] # Længde = 7
all_data_list = []


for i in range(0, len(hc_array)): # Laver 7 basis modeller der kan trænes og testes senere
    temp_all_data = AllData()
    gcn = GCN(in_features=dataset.num_node_features, hidden_channels= hc_array[i]) # Gemmer i array, da jeg er usikker på om python passer by reference eller value????
    gcn.to(device=device)
    base_model = BaseModel(gcn, loss_f, torch.optim.Adam(gcn.parameters(), lr=LEARNING_RATE)) 
    for epoch in trange(0, EPOCHS):
        temp_all_data.insert_train_data(train(base_model, train_loader, device))
        temp_all_data.insert_test_data(test(base_model, test_loader, device))
    

    all_data_list.append(temp_all_data)
  
MultiPlotter(all_data_list, hc_array, "Hidden node count")