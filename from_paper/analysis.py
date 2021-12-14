import pandas as pd
import os
import glob


path_this_file = os.path.dirname(os.path.abspath(__file__))
file_data = "fully-labeled-dataset.csv"
path_data = os.path.join(path_this_file,"..","data",file_data)
print(path_data)
df = pd.read_csv(path_data,sep="|")
print(df)
print(df.iloc[0,[0,1]])