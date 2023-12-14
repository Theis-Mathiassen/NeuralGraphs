import pandas as pd
import seaborn
import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.rcParams['figure.dpi'] = 400

def GetHeatData() :
    usecols = ["dropout_rate", "hidden_channels", "learning_rate", "batch_size","epochs","amount_of_layers","optimizer","activation_function","pooling_algorithm", "roc"]
    csv_data = pd.read_csv("results/CombinedNew.csv", usecols = usecols)

    #csv_data = csv_data.tail(3456*10)
    csv_data = pd.pivot_table(csv_data, values=usecols[9], index=['dropout_rate', 'batch_size', 'hidden_channels', 'amount_of_layers'], columns=[ 'epochs', 'learning_rate', 'activation_function', 'optimizer', 'pooling_algorithm'])

    csv_data.fillna(0, inplace=True) # Fills "NaN" values with 0 instead
    csv_data = csv_data.loc[:, (csv_data != 0).any(axis=0)] # Remove all 0-entry columns. If this problem exists with rows, change axis to 1 instead

    return csv_data


def heatMap():
    data = GetHeatData() # Gets data in the format that a clustermap desires


    # Assuming 'results' is a DataFrame with your data
    # Sample the data to avoid cluttering the heatmap
    #sample_size = min(1000, len(data))


    # Specify the columns you want to include in the heatmap
    columns_to_include = ['dropout_rate', 'hidden_channels', 'learning_rate', 'batch_size', 'epochs',
                        'amount_of_layers', 'optimizer', 'activation_function', 'pooling_algorithm',
                        'roc']
    csv_data = pd.read_csv("results/Grid_LastLastFr.csv")

    #subsampled_results = csv_data.sample(n=1000, replace=False)

    # Create the heatmap
    heatmap_data = csv_data[columns_to_include].pivot_table(aggfunc="mean", index=['optimizer', 'learning_rate'], columns=['pooling_algorithm', 'activation_function', 'amount_of_layers'], values='roc')
    print(heatmap_data)
    seaborn.set(font_scale=0.35)
    ax = seaborn.heatmap(heatmap_data, annot=True, cmap='viridis')

# Rotate the x-axis tick labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

# Rotate and align the y-axis tick labels
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va='center')
    

    plt.title('Heatmap: AUC-ROC')
    #plt.savefig(f'./results/heatmap.png', format='png', dpi=400)

    plt.show()

heatMap()
