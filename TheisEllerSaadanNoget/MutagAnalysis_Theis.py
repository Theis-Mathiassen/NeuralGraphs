import torch
import torch_geometric as TG
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
import networkx as nx
from torch_geometric.loader import DataLoader
from tqdm.auto import trange
import matplotlib.pyplot as plt
import numpy as np


from visualize import GraphVisualization
from MutagModel import GCN


if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    # MPS is currently slower than CPU due to missing int64 min/max ops
    device = torch.device('cpu')
else:
    device = torch.device('cpu')

#device = torch.device('cpu')



dataset_path = "data/TUDataset"
dataset = TUDataset(root=dataset_path, name='MUTAG')

dataset.download()



def create_graph(graph):
    g = to_networkx(graph)
    pos = nx.spring_layout(g)
    vis = GraphVisualization(
        g, pos, node_text_position='top left', node_size=20,
    )
    fig = vis.create_figure()
    return fig

def ShowAccuracies (accuracies):
    # x axis values
    x = np.array([i for i in range(len(accuracies))])
    # corresponding y axis values
    y = accuracies
    
    # plotting the points 
    plt.plot(x, y)
    
    # naming the x axis
    plt.xlabel('Epochs')
    # naming the y axis
    plt.ylabel('Accuracy')
    
    # giving a title to my graph
    plt.title('Training accuracy over time!')
    
    # function to show the plot
    plt.show()


torch.manual_seed(12345)
dataset = dataset.shuffle()
train_dataset = dataset[:150]
test_dataset = dataset[150:]

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')



train_loader = DataLoader(train_dataset, batch_size=75, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)

for step, data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(data)
    print()

print(torch.__version__)

print("Press 1 for old model, and other for new model.")
if (input() == "1"):
    print("You chose old model.")
    model = torch.load("graph_classification_model.pt")
else:
    print("You chose new model.")
    model = GCN(hidden_channels=32, dataset = dataset)

model.to(device=device)
print(model)





optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()



def train():
    model.train()
    loss_ = 0
    correct = 0
    i = 0
    for data in train_loader:  # Iterate in batches over the training dataset.
        data = data.to(device, non_blocking=True)
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss_ += loss.item()
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        
        
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        i+=1
    #print(len(train_loader.dataset))
    #print(f"correct: {correct}")
    train_loss.append(loss_/ i)
    train_accruracy.append(correct/len(train_loader.dataset))

def test(loader):
    model.eval()
    correct = 0
    loss_ = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        data.to(device)
        out = model(data.x, data.edge_index, data.batch)  
        loss = criterion(out, data.y)
        loss_ += loss.item()
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        
        
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    #print("Here")
    #print(len(loader.dataset))
    return correct / len(loader.dataset), loss_ / len(loader.dataset)  # Derive ratio of correct predictions.


iterations = 1000
train_accruracy = []
train_loss = []
test_acc = [0] * iterations
test_loss = [0] * iterations

for epoch in trange(0, iterations):
    train()


#testresult = test(train_loader)
#train_acc.append(testresult[0])
#train_loss = testresult[1] 
test_acc[epoch], test_loss[epoch] = test(test_loader)

torch.save(model, "graph_classification_model.pt")
    

# x axis values
x = np.array([i for i in range(iterations)])
# corresponding y axis values

print(len(x))
print(len(train_loss))

# plotting the points 

#print(train_accruracy)
plt.plot(x, train_accruracy, label="Accuracy")
plt.plot(x, train_loss, label="Train Loss")

plt.legend(loc="lower center")

print(test_acc[iterations-1])

# naming the x axis
plt.xlabel('Epochs')
# naming the y axis
plt.ylabel('Accuracy/Loss')

# giving a title to my graph
plt.title('Test over time!')

# function to show the plot
plt.show()



