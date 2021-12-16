import re
import pickle
import numpy as np
import pandas as pd

# Plot libraries
import seaborn as sns
# from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

#IMPORT DATASET
path_this_file = os.path.dirname(os.path.abspath(__file__))
file_data = "fully-labeled-dataset.csv"
cols=['id','tweet_text','text','emojis','sentiment_label','emoji_label','sarcasm_label']
path_data = os.path.join("d:\\NyeMan\\KULIAH S2\\Semester 1\\KK - B\\last-project","data",file_data)
df = pd.read_csv(path_data,sep="|")
# df = df[['text','emojis','sentiment_label','emoji_label','sarcasm_label']]
# print(df.head())

# ax = df.copy()[['text','sentiment_label']].groupby('sentiment_label').count().plot(kind='bar', title='Distribution of Sentiment Label',
#                                                legend=True)
# ax = ax.set_xticklabels(['Negative','Positive'], rotation=0)
# # # plt.show()

# ax = df.copy()[['emojis','emoji_label']].groupby('emoji_label').count().plot(kind='bar', title='Distribution of Emoji Label',
#                                                legend=True)
# ax = ax.set_xticklabels(['Negative','Positive','Neutral'], rotation=0)
# # plt.show()
# ax = df.copy()[['text','emojis','sarcasm_label']].groupby('sarcasm_label').count().plot(kind='bar', title='Distribution of Sarcasm Label',
#                                                legend=True)
# ax = ax.set_xticklabels(['Negative','Positive'], rotation=0)

# plt.show()
# print(df.copy()[['text','sentiment_label']].groupby('sentiment_label').count())
# print(df.copy()[['emojis','emoji_label']].groupby('emoji_label').count())
df_training = pd.DataFrame({'id':[],'tweet_text':[],'text':[],'emojis':[],'sentiment_label':[],'emoji_label':[],'sarcasm_label':[]})
df_testing = pd.DataFrame({'id':[],'tweet_text':[],'text':[],'emojis':[],'sentiment_label':[],'emoji_label':[],'sarcasm_label':[]})
grouped = df.groupby(['sentiment_label','sarcasm_label'])
pp_max = 103
pn_max = 1097
np_max = 1094
nn_max = 106
pp_i = 0
pn_i = 0
np_i = 0
nn_i = 0
for key,item in grouped:
  a_group = grouped.get_group(key)
  for index, row in a_group.iterrows():
      if(row['sentiment_label']==1 and row['sarcasm_label']==1): #pp
          if(pp_i < pp_max):
            pp_i+=1
            df_training = df_training.append(
                {'id':row["id"],
                'tweet_text':row["tweet_text"],
                'text':row['text'],
                'emojis':row["emojis"],
                'sentiment_label':row["sentiment_label"],
                'emoji_label':row["emoji_label"],
                'sarcasm_label':row["sarcasm_label"],
                }, ignore_index=True)
          else:
               df_testing = df_testing.append(
                {'id':row["id"],
                'tweet_text':row["tweet_text"],
                'text':row['text'],
                'emojis':row["emojis"],
                'sentiment_label':row["sentiment_label"],
                'emoji_label':row["emoji_label"],
                'sarcasm_label':row["sarcasm_label"],
                }, ignore_index=True)
      elif(row['sentiment_label']==1 and row['sarcasm_label']==0): #pn
          if(pn_i < pn_max):
            pn_i+=1
            df_training = df_training.append(
                {'id':row["id"],
                'tweet_text':row["tweet_text"],
                'text':row['text'],
                'emojis':row["emojis"],
                'sentiment_label':row["sentiment_label"],
                'emoji_label':row["emoji_label"],
                'sarcasm_label':row["sarcasm_label"],
                }, ignore_index=True)
          else:
               df_testing = df_testing.append(
                {'id':row["id"],
                'tweet_text':row["tweet_text"],
                'text':row['text'],
                'emojis':row["emojis"],
                'sentiment_label':row["sentiment_label"],
                'emoji_label':row["emoji_label"],
                'sarcasm_label':row["sarcasm_label"],
                }, ignore_index=True)
      elif(row['sentiment_label']==0 and row['sarcasm_label']==1): #np
          if(np_i < np_max):
            np_i+=1
            df_training = df_training.append(
                {'id':row["id"],
                'tweet_text':row["tweet_text"],
                'text':row['text'],
                'emojis':row["emojis"],
                'sentiment_label':row["sentiment_label"],
                'emoji_label':row["emoji_label"],
                'sarcasm_label':row["sarcasm_label"],
                }, ignore_index=True)
          else:
               df_testing = df_testing.append(
                {'id':row["id"],
                'tweet_text':row["tweet_text"],
                'text':row['text'],
                'emojis':row["emojis"],
                'sentiment_label':row["sentiment_label"],
                'emoji_label':row["emoji_label"],
                'sarcasm_label':row["sarcasm_label"],
                }, ignore_index=True)
      elif(row['sentiment_label']==0 and row['sarcasm_label']==0): #nn
          if(nn_i < nn_max):
            nn_i+=1
            df_training = df_training.append(
                {'id':row["id"],
                'tweet_text':row["tweet_text"],
                'text':row['text'],
                'emojis':row["emojis"],
                'sentiment_label':row["sentiment_label"],
                'emoji_label':row["emoji_label"],
                'sarcasm_label':row["sarcasm_label"],
                }, ignore_index=True)
          else:
               df_testing = df_testing.append(
                {'id':row["id"],
                'tweet_text':row["tweet_text"],
                'text':row['text'],
                'emojis':row["emojis"],
                'sentiment_label':row["sentiment_label"],
                'emoji_label':row["emoji_label"],
                'sarcasm_label':row["sarcasm_label"],
                }, ignore_index=True)
#   print(len(a_group))
#   print(a_group, "\n")
# print(df_training.head(), len(df_training))
# print(df_testing.head(), len(df_testing))

def getData():
    return df_training, df_testing