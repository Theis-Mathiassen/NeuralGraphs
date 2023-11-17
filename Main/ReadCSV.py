import pandas as pd
import seaborn as sns

def GetHeatData() :
    #"optimizer","activation_function","pooling_algorithm"
    usecols = ["dropout_rate", "hidden_channels", "learning_rate", "batch_size","epochs","amount_of_layers","roc"]
    csv_data = pd.read_csv("results/Combined_Grid.csv", usecols = usecols)
    print(csv_data)

    df = pd.DataFrame(csv_data, columns = usecols)
    print(df)
    
    return csv_data