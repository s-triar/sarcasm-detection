import os
import numpy as np
from gensim.models import Word2Vec
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# df_training, df_testing = ds.getData()

def predict_text_sentiment(X_test_text_dt):
    # X_test_text, y_test_text = np.array(df_testing['text']), np.array(df_testing['sentiment_label'])



    path_save_model = os.path.join("d:\\NyeMan\\KULIAH S2\\Semester 1\\KK - B\\last-project","models",'word2vec_model_text.model')
    word2vec_model_text = Word2Vec.load(path_save_model)
    # print(word2vec_model_text)

    input_length = 35
    vocab_length = 2400
    Embedding_dimensions = 100
    tokenizer_text=None



    path_save_model = os.path.join("d:\\NyeMan\\KULIAH S2\\Semester 1\\KK - B\\last-project","models",'tokenizer_text.model')
    with open(path_save_model, 'rb') as file:
        tokenizer_text = pickle.load(file)
    # print(tokenizer_text)



    X_test_text  = pad_sequences(tokenizer_text.texts_to_sequences(X_test_text_dt) , maxlen=input_length)

    # print("X_test_text.shape :", X_test_text.shape)


    embedding_matrix = np.zeros((vocab_length, Embedding_dimensions))

    for word, token in tokenizer_text.word_index.items():
        if word2vec_model_text.wv.__contains__(word):
            embedding_matrix[token] = word2vec_model_text.wv.__getitem__(word)

    # print("Embedding Matrix Shape:", embedding_matrix.shape)

    path_save_model = os.path.join("d:\\NyeMan\\KULIAH S2\\Semester 1\\KK - B\\last-project","models",'tf_text_data_paper_5')
    print("model:","tf_text_data_paper_5")
    text_sent_model = load_model(path_save_model)
    y_pred = text_sent_model.predict(X_test_text)
    # y_pred_round = np.where(y_pred>=0.5, 1, 0)
    # print(y_pred)
    return y_pred

# path_save_model = os.path.join("d:\\NyeMan\\KULIAH S2\\Semester 1\\KK - B\\last-project","models",'tf_text_data_paper_5')
# print("model:","tf_text_data_paper_5")
# model = load_model(path_save_model) 
# model.summary()
