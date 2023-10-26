import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import sklearn
from sklearn import metrics
from sklearn.metrics import RocCurveDisplay
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from datetime import datetime
from Classes import TrainData, TestData, AllData
import numpy as np


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


def GraphPrettifier(ax: axes.Axes, title, xlabel, ylabel):
    ax.set_title = title
    ax.set_xlabel = xlabel
    ax.set_ylabel = ylabel
    ax.axis('on')
    ax.set_facecolor('lightgray')
    ax.grid()
    ax.legend()

def PlotGraph(data, ax: axes.Axes, label, color = "black", cleanGraph = False, numChunks = 20):
    if cleanGraph and numChunks < len(data):
        X = np.arange(0, len(data), len(data)//numChunks)
        data = AvgCalculator(data, numChunks)
    else:
        X = np.arange(0, len(data))

    ax.plot(X, data, c=color, label=label)
    
def MultiPlotter(allData: list[AllData], paramArray, paramName):
    color_array = np.array(["lightsalmon", "navy", "goldenrod", "aquamarine", "rosybrown", "darkgoldenrod", "cyan", "forestgreen", "wheat", "springgreen", "mediumpurple", "violet", "deeppink", "burlywood", "powderblue", "indigo", "azure", "chocolate", "saddlebrown", "linen", "chartreuse", "black", "floralwhite", "cornflowerblue", "fuchsia", "magenta", "orchid", "hotpink", "lightblue", "lime", "navajowhite", "royalblue", "teal", "ivory", "sienna", "sandybrown"], dtype=str)



    fig, ax = plt.subplots(2,2, figsize=(14,5)) # Initialize figure
    i = 0
    for data in allData: # Plot training data accuracies & loss
        label = '{}: {}'.format(paramName, paramArray[i])
        PlotGraph(data.train_accuracies, ax[0,0], label, color = np.random.choice(color_array, replace=False), cleanGraph=True, numChunks=40)
        PlotGraph(data.train_losses, ax[0,1], label, color=np.random.choice(color_array, replace=False), cleanGraph=True, numChunks=40)
        #PlotGraph(data.test_accuracies, ax[1,0], label, color=np.random.choice(color_array, replace=False), cleanGraph=True, numChunks=40)
        i += 1
    GraphPrettifier(ax[0,0], "Training: Accuracy over epochs", "# Epochs", "Accuracy")
    GraphPrettifier(ax[0,1], "Training: Losses over epochs", "# Epochs", "Losses")
    

    now = datetime.now()

    title = "Precision measurements using different values for {}. Profiled on {}".format(paramName, now.strftime("%d/%m/%y), %H:%M:%S"))
    fig.suptitle(title, fontsize=12, wrap=True)
    fig.show()

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

def AndreasPlot(trainData: TrainData, testData: TestData, MANUAL_SEED, DATASPLIT, EPOCHS, LEARNING_RATE):
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

            plotAccuracy(trainData.train_losses, trainData.train_accuracies, "Training: Accuracy & Loss", ax00)

        if(plot_rocauc):
            #ax10 = ax[0, 1]
            ax10 = plt.subplot(gs[2, 0])

            plotROCAUC(trainData.train_labels, trainData.train_scores, "Training: ROC AUC", ax10)

    # Plot for test
    if(plot_testing):
        if(plot_accuracy):
            #ax01 = ax[1, 0]
            ax01 = plt.subplot(gs[1, 1])
            plotAccuracy(testData.test_losses, testData.test_accuracies, "Testing: Accuracy & Loss", ax01)

        if(plot_rocauc):
            #ax11 = ax[1, 1]
            ax11 = plt.subplot(gs[2, 1])

            plotROCAUC(testData.test_labels, testData.test_scores, "Testing: ROC AUC" ,ax11)

    now = datetime.now()

    title = "Model algorithm/parameter & performance profile. Profiled on {}.".format(now.strftime("%d/%m/%Y, %H:%M:%S"))

    fig.suptitle(title, fontsize=12, wrap=True)

    plt.tight_layout()