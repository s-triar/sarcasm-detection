import pandas as pd
import os

path_this_file = os.path.dirname(os.path.abspath(__file__))
file_data = "colloqial-lexicon.csv"
path_data = os.path.join("d:\\NyeMan\\KULIAH S2\\Semester 1\\KK - B\\last-project","data",file_data)
# print(path_data)
df = pd.read_csv(path_data,sep=",")

# print(df['slang'])
# hasil = df[df['slang']=="aaa"]
# print(hasil)
# print(len(hasil))
# print(hasil['formal'].values[0])

def check_word(word):
    hasil = df[df['slang']==word]
    if(len(hasil)==0):
        return ""
    else:
        return hasil['formal'].values[0]
    