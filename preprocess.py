import pandas as pd
import os
import glob
from libs import req_to_tesaurus_indo
from libs import check_colloqial
# from libs import req_to_kbbi
import re
import csv

path_this_file = os.path.dirname(os.path.abspath(__file__))
file_data = "fully-labeled-dataset.csv"
path_data = os.path.join("d:\\NyeMan\\KULIAH S2\\Semester 1\\KK - B\\last-project","data",file_data)
df = pd.read_csv(path_data,sep="|")
path_save = os.path.join("d:\\NyeMan\\KULIAH S2\\Semester 1\\KK - B\\last-project","data","tw_to_mine_no_kbbi.csv")
new_df = pd.DataFrame({'id':[],'tweet_text':[],'text':[],'emojis':[],'sentiment_label':[],'emoji_label':[],'sarcasm_label':[]})
cols=['id','tweet_text','text','emojis','sentiment_label','emoji_label','sarcasm_label']
for index, row in df[3072:].iterrows():
    print(row["id"], row["tweet_text"])
    raw_text = row['tweet_text']
    clean_txt = ' '.join(re.sub("([@#][A-Za-z0-9_]+)|([^0-9A-Za-z\- \t])|(\w+:\/\/\S+)"," ",raw_text).split())
    clean_txt = clean_txt.lower()
    words = clean_txt.split()
    new_words = []
    for w in words:
        # hasil_cek_cq = check_colloqial.check_word(w)
        # if(hasil_cek_cq!=""):
        #     hasil_cek = req_to_tesaurus_indo.check_with_tesaurus(hasil_cek_cq)
        #     if(hasil_cek):
        #         new_words.append(hasil_cek_cq)
        # else:
        #     hasil_cek = req_to_tesaurus_indo.check_with_tesaurus(w)
        #     if(hasil_cek):
        #         new_words.append(w)
        
        hasil_cek = req_to_tesaurus_indo.check_with_tesaurus(w)
        if(hasil_cek):
            # ada di tesaurus
            new_words.append(w)
        else:
            hasil_cek_cq = check_colloqial.check_word(w)
            if(hasil_cek_cq!=""):
                new_words.append(hasil_cek_cq)
            # else:
            #     hasil_cek = req_to_kbbi.check_with_kkbi(w)
            #     if(hasil_cek):
            #         # ada di kamus
            #         new_words.append(w)
            
    clean_sentence = ' '.join(new_words)
    # print(clean_sentence)
    # new_df = new_df.append(
    #     {'id':row["id"],
    #      'tweet_text':row["tweet_text"],
    #      'text':clean_sentence,
    #      'emojis':row["emojis"],
    #      'sentiment_label':row["sentiment_label"],
    #      'emoji_label':row["emoji_label"],
    #      'sarcasm_label':row["sarcasm_label"],
    #     }, ignore_index=True)
    print(clean_sentence)
    data = {'id':row["id"],
        'tweet_text':row["tweet_text"],
        'text':clean_sentence,
        'emojis':row["emojis"],
        'sentiment_label':row["sentiment_label"],
        'emoji_label':row["emoji_label"],
        'sarcasm_label':row["sarcasm_label"],
        }
    with open(path_save, 'a+', encoding='utf8',newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=cols,delimiter="|")
        writer.writerow(data)
# new_df.to_csv(path_save, index=False, sep='|')