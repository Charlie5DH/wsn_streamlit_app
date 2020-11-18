import numpy as np
import pandas as pd

def get_module_feature(module, features, path_of_file, numerical=False):
    
    df = pd.read_csv(path_of_file, parse_dates=['Timestamp'], index_col='Timestamp')
    
    try:
        df = df.drop(['Unnamed: 0'],axis=1)
    except:
        pass
    if numerical:
        return np.array(df[df['Module']==module][[features]])
    else:
        df[df['Module']==module][[features]]

