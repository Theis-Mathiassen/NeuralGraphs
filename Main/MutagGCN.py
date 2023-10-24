EPOCHS = 300
LEARNING_RATE = 0.01
MANUAL_SEED = 12345
HIDDEN_NODE_COUNT = 64
DATASPLIT = 150
BATCH_SIZE_TRAINING = 64
BATCH_SIZE_TESTING = 1


import os
import torch
import torch_geometric

from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
from torch_geometric.loader import DataLoader



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

import networkx as nx
from visualize import GraphVisualization

def create_graph(graph):
    g = to_networkx(graph)
    pos = nx.spring_layout(g)
    vis = GraphVisualization(
        g,
        pos,
        node_text_position="top left",
        node_size=20,
    )
    fig = vis.create_figure()
    return fig


fig = create_graph(dataset[0])
fig.show()






from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


class GCN(torch.nn.Module):
    def __init__(self, in_features=dataset.num_node_features, hidden_channels=HIDDEN_NODE_COUNT, outfeatures = 2):
        super(GCN, self).__init__()

        # Input layer
        self.conv1 = GCNConv(in_features, hidden_channels)

        # Hidden layers
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        #self.conv3 = GCNConv(hidden_channels, hidden_channels)
        #self.conv4 = GCNConv(hidden_channels, hidden_channels)
        #self.conv5 = GCNConv(hidden_channels, hidden_channels)

        # Output layer
        self.lin = Linear(hidden_channels, outfeatures)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        #x = self.conv3(x, edge_index)
        #x = x.relu()
        #x = self.conv4(x, edge_index)
        #x = x.relu()
        #x = self.conv5(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x
    

class GCNData():
    def __init__(self):
        self.train_losses = []
        self.train_accuracies = []
        self.train_labels = []
        self.train_scores = []
        self.test_losses = []
        self.test_accuracies = []
        self.test_scores = []
        self.test_labels = []


from tqdm import trange

loss_function = torch.nn.CrossEntropyLoss()

def hyperParameterTester(create_table=False, hc=HIDDEN_NODE_COUNT, learn=LEARNING_RATE):
    
    torch.manual_seed(MANUAL_SEED)                              # Set manual seed
    model = GCN(hidden_channels=hc)                             # Initilialize model
    model.to(device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learn)  # Set optimizer with learning rate
    loss_function = torch.nn.CrossEntropyLoss()                 # Set loss function

    returnData = GCNData()

    def train():
        model.train()
        loss_ = 0
        correct = 0
        i = 0

        for data in train_loader:  # Iterates the batches. We declared each batch to be of size 64
            data = data.to(device, non_blocking=True)
            # Calculate output, and get the maximum of those in order to obtain the predicted value
            out = model(data.x, data.edge_index, data.batch)
            cat = torch.argmax(out, dim=1)

            correct += int((cat == data.y).sum())  # Check against ground-truth labels.
            
            loss = loss_function(out, data.y)
            loss_ += loss.item()
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            i+=1

            # Append actual and preddicted to respective array. Have to be converted to NumPy arrays in order to flatten them.
            # We flatten them as 1D arrays are required by SK in order to calculate and plot ROC AUC
            #arrayLabel = np.array(data.y.to('cpu'))
            #for value in arrayLabel.flatten():
            #    returnData.train_labels.append(value)

            #arrayCat = np.array(cat.to('cpu'))
            #for value in arrayCat.flatten():
            #    returnData.train_scores.append(value)
        
        #tt.set_description("loss: %2f. accuracy %2f." % (loss, correct/len(train_loader.dataset)))
        returnData.train_losses.append(loss_/ i)
        returnData.train_accuracies.append(correct/len(train_loader.dataset))

    def test(loader):
        model.eval()
        correct = 0
        loss_ = 0
        i = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            data = data.to(device, non_blocking=True)
            out = model(data.x, data.edge_index, data.batch)
            cat = torch.argmax(out, dim=1)

            correct += int((cat == data.y).sum())  # Check against ground-truth labels.

            loss = loss_function(out, data.y)
            loss_ += loss.item()

            arrayLabel = np.array(data.y.to('cpu'))
            for value in arrayLabel.flatten():
                returnData.test_labels.append(value)

            arrayCat = np.array(cat.to('cpu'))
            for value in arrayCat.flatten():
                returnData.test_scores.append(value)
            
            i+=1
        
        returnData.test_losses.append(loss_/ i)
        returnData.test_accuracies.append(correct/len(loader.dataset))

    for epoch in trange(1, EPOCHS):
        train()
    test(train_loader)
    
    return returnData

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.style as mplstyle
import sklearn
from sklearn import metrics
from sklearn.metrics import RocCurveDisplay

def GraphPrettifier(figname, ylabel, title):
    figure = plt.figure(figname)
    figure.patch.set_facecolor('lightgray')
    plt.xlabel("# Epochs")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.legend()

# Used for prettier graph
def AvgCalculator(data, numChunks):
    averageOfData = []
    chunkSize = len(data) // numChunks
    for i in range(0, len(data), chunkSize):
        chunk = data[i:i+chunkSize]
        chunkAvg = sum(chunk) / len(chunk)
        averageOfData.append(chunkAvg)
    return averageOfData

# ROC AUC PLOT
def plotROCAUC(labels, scores, title, ax):
    # roc_auc = roc_auc_score(labels, scores)

    fpr, tpr, _ = metrics.roc_curve(labels,  scores)
    auc = metrics.roc_auc_score(labels, scores)
    ax.set_title(title)
    ax.plot(fpr,tpr)
    ax.legend(["auc="+str(round(auc, 2))], handlelength=0, handletextpad=0)

    # RocCurveDisplay.from_predictions(labels, scores)

    #plt.show()


class PP():
    def __init__(self, value, colour):
        self.value = value
        self.colour = colour


def PlotBasic():
    data = hyperParameterTester() # Get data
    AndreasPlot(data)
    X = np.arange(0, len(data.train_accuracies))
    plt.figure('g', figsize=[10,5])
    plt.plot(X, np.array(data.train_accuracies), c="indianred", label="Accuracy")
    plt.plot(X, np.array(data.train_losses), c="navy", label="Loss")
    GraphPrettifier('g', '', 'Accuracy and Loss')
    plt.show()




def PlotOverLR(para: GCNData, cleanGraph=False, numChunks = 50):
    EstNumData = (EPOCHS-1)*len(train_loader)
    X = np.arange(0, (EPOCHS-1)*len(train_loader)) if not cleanGraph else np.arange(0, EstNumData, EstNumData//numChunks)

    for p in para:
        data = hyperParameterTester(learn = p.value) # Get data

        plt.figure('Acc', figsize=[10,5]) # Activate figure with accuracy
        accuracies = np.array(data.train_accuracies)
        if not cleanGraph: plt.plot(X, accuracies, color=p.colour, label='Learning rate: {}'.format(p.value))

        plt.figure('Loss', figsize=[10,5]) # Activate figure with loss
        losses = np.array(data.train_losses)
        if not cleanGraph: plt.plot(X, losses, color=p.colour, label='Learning rate: {}'.format(p.value))

        if cleanGraph: # Taking averages for prettier graph
            averageAcc = AvgCalculator(accuracies, numChunks)
            averageLoss = AvgCalculator(losses, numChunks)
            
            plt.figure('Acc') # Activate figure with accuracy
            plt.plot(X, averageAcc, color=p.colour, label='Learning rate: {}'.format(p.value))
            plt.figure('Loss') # Activate figure with loss
            plt.plot(X, averageLoss, color=p.colour, label='Learning rate: {}'.format(p.value))

    script_dir = os.getcwd()
    save_dir = os.path.join(script_dir, 'Figures/LR/')

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    save_name = "Accuracy.pdf"
    GraphPrettifier('Acc', 'Accuracy', 'Accuracy with different learning rates')
    plt.savefig(save_dir + save_name)

    save_name = "Loss.pdf"
    GraphPrettifier('Loss', 'Loss', 'Loss with different learning rates')
    plt.savefig(save_dir + save_name)

    plt.show()


def PlotOverHC(para: GCNData, cleanGraph=False, numChunks = 50):
    EstNumData = (EPOCHS-1)*len(train_loader)
    X = np.arange(0, (EPOCHS-1)*len(train_loader)) if not cleanGraph else np.arange(0, EstNumData, EstNumData//numChunks)

    for p in para:
        data = hyperParameterTester(hc = p.value) # Get data

        plt.figure('Acc', figsize=[10,5]) # Activate figure with accuracy
        accuracies = np.array(data.train_accuracies)
        if not cleanGraph: plt.plot(X, accuracies, color=p.colour, label='# Neurons: {}'.format(p.value))

        plt.figure('Loss', figsize=[10,5]) # Activate figure with loss
        losses = np.array(data.train_losses)
        if not cleanGraph: plt.plot(X, losses, color=p.colour, label='# Neurons: {}'.format(p.value))

        if cleanGraph: # Taking averages for prettier graph
            averageAcc = AvgCalculator(accuracies, numChunks)
            averageLoss = AvgCalculator(losses, numChunks)
            
            plt.figure('Acc') # Activate figure with accuracy
            plt.plot(X, averageAcc, color=p.colour, label='# Neurons: {}'.format(p.value))
            plt.figure('Loss') # Activate figure with loss
            plt.plot(X, averageLoss, color=p.colour, label='# Neurons: {}'.format(p.value))

    script_dir = os.getcwd()
    save_dir = os.path.join(script_dir, 'Figures/HC/')

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    save_name = "Accuracy.pdf"
    GraphPrettifier('Acc', 'Accuracy', 'Accuracy with different neurons in the hidden layers')
    plt.savefig(save_dir + save_name)

    save_name = "Loss.pdf"
    GraphPrettifier('Loss', 'Loss', 'Loss with different neurons in the hidden layers')
    plt.savefig(save_dir + save_name)

    plt.show()


import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from datetime import datetime

# ACCURACY PLOT
def plotAccuracy(losses, accuracies, title, ax):
    #fig, ax[0,1] = sub
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.plot(losses)
    ax.plot(accuracies)
    ax.plot(losses, label="Loss")
    ax.plot(accuracies, label="Accuracy")
    ax.legend(loc="lower center")
    
    
    #plt.show()

def AndreasPlot(data: GCNData):
    plot_training = True
    plot_testing = True

    plot_accuracy = True
    plot_rocauc = True

    fig = plt.figure(figsize=(8, 6))

    gs = gridspec.GridSpec(3, 2)

    ax1 = plt.subplot(gs[0, :])



    table_data = [
        ["Seed", "Activation", "Weight Initializer", "Loss function", "Pooling", "Optimizer", "# GCN layers", "Neurons", "Split", "Epochs", "Batch Size", "LR"],
            [str(MANUAL_SEED), "ReLU", "Default", "C-Entropy", "GMax Pooling", "Adam", "3", "7, 64, 64, 64, 2", str(round(DATASPLIT/188, 2)), str(EPOCHS), "64", str(LEARNING_RATE)],
    ]

    table_data_1 = [table_data[0][:len(table_data[0]) // 2], table_data[1][:len(table_data[1]) // 2]]
    table_data_2 = [table_data[0][len(table_data[0]) // 2:], table_data[1][len(table_data[1]) // 2:]]

    # Create the first table in the top subplot (upper section)
    table_1 = ax1.table(cellText=table_data_1, cellLoc='center', loc='center')
    table_1.auto_set_font_size(False)
    table_1.set_fontsize(12)
    table_1.scale(1.2, 1.2)

    # Create the second table in the top subplot (lower section)
    table_2 = ax1.table(cellText=table_data_2, cellLoc='center', loc='bottom')
    table_2.auto_set_font_size(False)
    table_2.set_fontsize(12)
    table_2.scale(1.2, 1.2)



    # Hide axis and display the table
    #ax1 = plt.gca()
    ax1.axis('off')

    #divider_y = -1.1  # Adjust the y-coordinate as needed
    #ax1.axhline(divider_y, color='black')




    # Plot for train
    if(plot_training):
        if(plot_accuracy):
            #ax00 = ax[0, 0]
            ax00 = plt.subplot(gs[1, 0])

            plotAccuracy(data.train_losses, data.train_accuracies, "Training: Accuracy & Loss", ax00)

        if(plot_rocauc):
            #ax10 = ax[0, 1]
            ax10 = plt.subplot(gs[2, 0])

            plotROCAUC(data.train_labels, data.train_scores, "Training: ROC AUC", ax10)

    # Plot for test
    if(plot_testing):
        if(plot_accuracy):
            #ax01 = ax[1, 0]
            ax01 = plt.subplot(gs[1, 1])
            plotAccuracy(data.test_losses, data.test_accuracies, "Testing: Accuracy & Loss", ax01)

        if(plot_rocauc):
            #ax11 = ax[1, 1]
            ax11 = plt.subplot(gs[2, 1])

            plotROCAUC(data.test_labels, data.test_scores, "Testing: ROC AUC" ,ax11)

    now = datetime.now()

    title = "Model algorithm/parameter & performance profile. Profiled on {}.".format(now.strftime("%d/%m/%Y, %H:%M:%S"))

    fig.suptitle(title, fontsize=12, wrap=True)

    plt.tight_layout()



LRparameters = {PP(0.1, 'lightsalmon'), PP(0.01, 'goldenrod'), PP(0.001, 'mediumaquamarine'), PP(0.005, 'rosybrown')}
HCparameters = {PP(5, 'firebrick'), PP(10, 'lightsalmon'), PP(20, 'darkgoldenrod'), PP(40, 'goldenrod'), PP(64, 'mediumaquamarine'), PP(100, 'black'), PP(200, 'rosybrown')}
#PlotOverLR(LRparameters, cleanGraph=True)
#PlotOverHC(HCparameters, cleanGraph=True)
PlotBasic()