import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, accuracy_score
from libs import data_selection as ds
import text_sentiment as ts
import emoji_sentiment as es

df_training, df_testing = ds.getData()
X_test_text, y_test_text = np.array(df_testing['text']), np.array(df_testing['sentiment_label'])
X_test_emoji, y_test_emoji = np.array(df_testing['emojis']), np.array(df_testing['emoji_label'])
X_test_sarc, y_test_sarcasm = np.array(df_testing['text']), np.array(df_testing['sarcasm_label'])



y_labels=y_test_sarcasm.flatten()
threshold = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
text_sentiment_scores= (ts.predict_text_sentiment(X_test_text)).flatten()
print(text_sentiment_scores)
emoji_sentiment_scores= es.get_pre_predict()#predict_emoji_sentiment(X_test_emoji)
print(emoji_sentiment_scores)
predicts = []
for constant in threshold:
    for text, emo in zip(text_sentiment_scores, emoji_sentiment_scores):
        # print(constant, text,emo)
        distance = abs(text-emo)
        if(distance>constant):
            predicts.append(1)
        else:
            predicts.append(0)
    predicts=np.array(predicts)
    cf_matrix = confusion_matrix(y_labels, predicts)
    clas_report = classification_report(y_labels, predicts)
    print("thres", constant)
    print("Confusion matrix")
    print("precision", round(precision_score(y_labels,predicts),4))
    print("recall", round(recall_score(y_labels,predicts),4))
    print("f1 score", round(f1_score(y_labels,predicts),4))
    print("accuracy", round(accuracy_score(y_labels,predicts),4))
    print("Classification report")
    print(clas_report)
    print(predicts)
    predicts=[]