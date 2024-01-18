import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import sklearn
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import RocCurveDisplay
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from datetime import datetime
from Classes import TrainData, TestData, AllData
import numpy as np
from random import randint
import matplotlib.cm as cm
import pandas as df
from ReadCSV import GetHistData




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
def plotROCAUC(ax, labels, probabilities, label, color):
    # roc_auc = roc_auc_score(labels, scores)

    fpr, tpr, _ = metrics.roc_curve(labels,  probabilities)
    auc = metrics.roc_auc_score(labels, probabilities)
    ax.plot(fpr,tpr, c=color, label=label+"- AUC: {}".format(round(auc, 2)))

    # RocCurveDisplay.from_predictions(labels, scores)

    #plt.show()


def GraphPrettifier(ax: axes.Axes, title, xlabel, ylabel):
    ax.title.set_text(title)
    ax.set_xlabel(xlabel=xlabel, fontsize=10)
    ax.set_ylabel(ylabel=ylabel, fontsize=10)
    ax.axis('on')
    ax.set_facecolor('lightgray')
    ax.grid()
    legend = ax.legend()
    legend.get_frame().set_alpha(None)
    legend.get_frame().set_facecolor((0,0,1,0.1))

    


def PlotGraph(data, ax: axes.Axes, label, color = "black", cleanGraph = False, numChunks = 20):
    if cleanGraph and numChunks < len(data):
        X = np.arange(0, len(data), len(data)//numChunks)
        data = AvgCalculator(data, numChunks)
    else:
        X = np.arange(0, len(data))

    ax.plot(X, data, c=color, label=label)
    
def MultiPlotter(allData: list[AllData], paramArray, paramName):
    color_array = np.array(["lightsalmon", "navy", "azure", "aquamarine", "rosybrown", "cyan", "forestgreen", "wheat", "springgreen", "mediumpurple", "violet", "deeppink", "burlywood", "powderblue", "indigo", "azure", "chocolate", "saddlebrown", "linen", "chartreuse", "black", "floralwhite", "cornflowerblue", "fuchsia", "magenta", "orchid", "hotpink", "lightblue", "lime", "navajowhite", "royalblue", "teal", "ivory", "sienna", "sandybrown"], dtype=str)

    fig, ax = plt.subplots(3,2, figsize=(21,10)) # Initialize figure
    i = 0
    for data in allData: # Plot training data accuracies & loss
        label = '{}: {}'.format(paramName, paramArray[i])
        PlotGraph(data.train_accuracies, ax[0,0], label, color=color_array[i], cleanGraph=True, numChunks=20)
        PlotGraph(data.test_accuracies, ax[0,1], label, color=color_array[i], cleanGraph=True, numChunks=20)
        PlotGraph(data.train_losses, ax[1,0], label, color=color_array[i], cleanGraph=True, numChunks=20)
        PlotGraph(data.test_losses, ax[1,1], label, color=color_array[i], cleanGraph=True, numChunks=20)
        plotROCAUC(ax[2,0], data.train_labels, data.train_probability_estimates, label, color=color_array[i])
        plotROCAUC(ax[2,1], data.test_labels, data.test_probability_estimates, label, color=color_array[i])
        i += 1
    GraphPrettifier(ax[0,0], "Training: Accuracy over epochs", "# Epochs", "Accuracy")
    GraphPrettifier(ax[0,1], "Testing: Accuracy over epochs", "# Epochs", "Accuracy")
    GraphPrettifier(ax[1,0], "Training: Loss", "# Epochs", "Loss")
    GraphPrettifier(ax[1,1], "Testing: Loss", "# Epochs", "Loss")
    GraphPrettifier(ax[2,0], "Training: AUC ROC", "True Negative", "True Positive")
    GraphPrettifier(ax[2,1], "Testing: AUC ROC", "True Negative", "True Positive")

    

    now = datetime.now()

    title = "Precision measurements using different values for {}. Profiled on {}".format(paramName, now.strftime("%d/%m/%y), %H:%M:%S"))
    fig.suptitle(title, fontsize=12, wrap=True)
    fig.tight_layout(pad=2.0)

    plt.show()

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


def HyperParamSearchPlot(test_scores, eval_metric : str) :
    test_scores = np.array(test_scores)
    #test_scores.sort()
    X = np.linspace(0, test_scores.size, num=test_scores.size)
    ylim = np.max(test_scores) * 1.1
    xmax = test_scores.argmax()
    random_color="#" + f"{randint(0, 0xFFFFFF):06x}"
    
    plt.figure()
    ax=plt.axes()
    ax.set_facecolor('lightgray')
    plt.scatter(X, test_scores, s = 1.5, c=test_scores, cmap='inferno')
    plt.colorbar()
    plt.ylim(0, ylim)
    plt.annotate('Max', xy=(xmax, ylim), xytext=(xmax-0.1, ylim+0.13), arrowprops = dict(facecolor='black', shrink=0.015))
    plt.grid()

    plt.xlabel('# Permutation')
    plt.ylabel(eval_metric)
    plt.title(eval_metric + " over permutations")
    plt.show()

def HeatMap(data) :
    map = sns.clustermap(data, cmap='magma', vmin=0, vmax=1, metric='correlation', z_score=None, standard_scale=None, yticklabels=True, figsize=(12, 8))

    plt.show()

def GridBayesianComparison(gridParam, bayesParam, gridVal, bayesVal, parameter):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figheight(5)
    fig.set_figwidth(10)
    fig.suptitle("Comparing Grid Search and Bayesian Optimization Over Iterations")

    X = np.linspace(0, len(gridParam), len(gridParam))

    ax1.scatter(X, gridParam, s=2.5, label='Grid', c='blue')
    ax1.scatter(X, bayesParam, s=2.5, label='Bayesian', c='red')
    ax1.set_xlabel('Number of iterations')
    ax1.set_ylabel('Number of ' + parameter)
    ax1.set_facecolor('lightgray')
    ax1.legend(loc='upper left')  # Adjust the location as needed
    ax1.set_title('Epochs over iterations')  # Set the subplot title
    ax1.grid()

    ax2.scatter(X, gridVal, s=3.5, label='Grid', c='blue')
    ax2.scatter(X, bayesVal, s=3.5, label='Bayesian', c='red')
    ax2.set_xlabel('Number of iterations')
    ax2.set_ylabel('AUC-ROC Value')
    ax2.set_facecolor('lightgray')
    ax2.legend(loc='lower left')  # Adjust the location as needed
    ax2.set_title('AUC-ROC value over iterations')  # Set the subplot title
    ax2.grid()

    plt.savefig('comparison')
    plt.tight_layout()  # Adjust subplot parameters for better layout
    plt.show()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, FuncFormatter

# Assume you have GetHistData and MakeHist functions defined

def GridBayesHist():
    def format_y_axis(value, pos):
        return f"{int(value):,}"  # Format y-axis as integer with commas

    fig1, ax1 = plt.subplots(3, 2, constrained_layout=True, figsize=(14, 8))
    fig1.suptitle('Grid Search Development Over Iterations', fontsize=16)

    fig2, ax2 = plt.subplots(3, 2, constrained_layout=True, figsize=(14, 8))
    fig2.suptitle('Bayesian Optimization Development Over Iterations', fontsize=16)

    i = 0
    j = 0

    for count in [2, 10, 20, 50, 100, 500]:
        grid_data, bayes_data = GetHistData('roc', count)
        index1 = int(np.floor(i))
        index2 = j % 2
        MakeHist(grid_data, bayes_data, ax1[index1][index2], ax2[index1][index2])
        i += 1 / 2
        j += 1

    fig1.subplots_adjust(left=0.045, bottom=0.048, right=0.971, top=0.9, wspace=0.2, hspace=0.3)
    fig2.subplots_adjust(left=0.045, bottom=0.048, right=0.971, top=0.9, wspace=0.2, hspace=0.3)

    for ax_row in ax1:
        for ax in ax_row:
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # Choose maximum number of ticks for y-axis
            ax.yaxis.set_major_formatter(FuncFormatter(format_y_axis))  # Apply custom formatter to y-axis labels
            ax.tick_params(axis='both', labelsize=8)  # Adjust label font size for better readability

    for ax_row in ax2:
        for ax in ax_row:
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # Choose maximum number of ticks for y-axis
            ax.yaxis.set_major_formatter(FuncFormatter(format_y_axis))  # Apply custom formatter to y-axis labels
            ax.tick_params(axis='both', labelsize=8)  # Adjust label font size for better readability

    fig1.savefig('fig1.png')
    fig2.savefig('fig2.png')
    plt.show()



def MakeHist(gridROC, bayesROC, axgrid : axes,  axbayes : axes) : 
    counts1, bins1 = np.histogram(gridROC, bins=20)
    counts2, bins2 = np.histogram(bayesROC, bins=20)
    print(counts2, bins2)

    axgrid.hist(bins1[:-1, ], bins1, weights=counts1, label=f"{len(gridROC)} iterations")
    axgrid.legend()
    axgrid.set_xlabel('Eval score')
    axgrid.set_ylabel('Count')
    axgrid.set_facecolor('lightgray')

    axbayes.hist(bins2[:-1, ], bins2, weights=counts2, label=f"{len(bayesROC)} iterations")
    axbayes.legend()
    axbayes.set_xlabel('Eval score')
    axbayes.set_ylabel('Count')
    axbayes.set_facecolor('lightgray')


