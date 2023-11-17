import pandas as pd
import seaborn as sns

def GetHeatData() :
    usecols = ["dropout_rate", "hidden_channels", "learning_rate", "batch_size","epochs","amount_of_layers","optimizer","activation_function","pooling_algorithm", "roc"]
    csv_data = pd.read_csv("results/Combined_Grid.csv", usecols = usecols)

    csv_data = csv_data.head(3456) #Currently necessary 

    csv_data = pd.pivot_table(csv_data, values='roc', index=['dropout_rate', 'batch_size', 'hidden_channels', 'amount_of_layers'], columns=[ 'epochs', 'learning_rate', 'activation_function', 'optimizer', 'pooling_algorithm'])

    return csv_data