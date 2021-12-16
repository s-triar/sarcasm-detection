# import tensorflow_datasets as tfds
# import tensorflow as tf

# tfds.disable_progress_bar()

# Utilities
import re
import pickle
import numpy as np
import pandas as pd

# Plot libraries
import seaborn as sns
# from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
from libs import data_selection as ds

df_training, df_testing = ds.getData()

X_train_text, y_train_text = np.array(df_training['text']), np.array(df_training['sentiment_label'])
X_test_text, y_test_text = np.array(df_testing['text']), np.array(df_testing['sentiment_label'])


# WORD EMBEDDING
from gensim.models import Word2Vec

Embedding_dimensions = 100

# Creating Word2Vec training dataset.
Word2vec_train_data_text = list(map(lambda x: x.split(), X_train_text))


# Defining the model and training it.
word2vec_model_text = Word2Vec(Word2vec_train_data_text,
                 vector_size=Embedding_dimensions,
                 workers=8,
                 window=5,
                 alpha=0.05,
                 epochs=30,
                 min_count=5)
path_save_model = os.path.join("d:\\NyeMan\\KULIAH S2\\Semester 1\\KK - B\\last-project","models",'word2vec_model_text.model')
word2vec_model_text.save(path_save_model)
print("Vocabulary Length:", len(word2vec_model_text.wv.key_to_index))

# Defining the model input length.
input_length = 35

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
vocab_length = 2400

tokenizer_text = Tokenizer(filters="", lower=False, oov_token="<oov>")
tokenizer_text.fit_on_texts(X_train_text)
tokenizer_text.num_words = vocab_length
print("Tokenizer Text vocab length:", tokenizer_text.num_words)

# Saving the tokenizer
path_save_model = os.path.join("d:\\NyeMan\\KULIAH S2\\Semester 1\\KK - B\\last-project","models",'tokenizer_text.model')
with open(path_save_model, 'wb') as file:
    pickle.dump(tokenizer_text, file)
    

# FOR TEXT only
X_train_text = pad_sequences(tokenizer_text.texts_to_sequences(X_train_text), maxlen=input_length)
X_test_text  = pad_sequences(tokenizer_text.texts_to_sequences(X_test_text) , maxlen=input_length)

print("X_train_text.shape:", X_train_text.shape)
print("X_test_text.shape :", X_test_text.shape)

embedding_matrix = np.zeros((vocab_length, Embedding_dimensions))

for word, token in tokenizer_text.word_index.items():
    if word2vec_model_text.wv.__contains__(word):
        embedding_matrix[token] = word2vec_model_text.wv.__getitem__(word)

print("Embedding Matrix Shape:", embedding_matrix.shape)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D, Dense, LSTM, Conv1D, Embedding, MaxPooling1D

def getModel():
    embedding_layer = Embedding(input_dim = vocab_length,
                                output_dim = Embedding_dimensions,
                                weights=[embedding_matrix],
                                input_length=input_length,
                                trainable=True)

    model = Sequential([
        embedding_layer,
        Bidirectional(LSTM(32, dropout=0.3, return_sequences=True)),
        # Bidirectional(LSTM(128, dropout=0.3, return_sequences=True)),
        # Bidirectional(LSTM(32, dropout=0.3, return_sequences=True)),
        
        # Conv1D(16, 5, activation='relu'),
        Conv1D(8, 5, activation='relu'),
        # MaxPooling1D(pool_size=2),
        GlobalMaxPool1D(),
        Dense(2, activation='relu'),
        Dense(1, activation='sigmoid'),
    ],
    name="Sentiment_Model")
    return model

training_model = getModel()
training_model.summary()
print(X_train_text.shape, y_train_text.shape)
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

path_save_model_cp = os.path.join("d:\\NyeMan\\KULIAH S2\\Semester 1\\KK - B\\last-project","models",'tf_text_weight_cp_data_paper_5')

callbacks = [
            ReduceLROnPlateau(monitor='val_loss', patience=50, cooldown=0),
            EarlyStopping(monitor='val_accuracy', min_delta=1e-4, patience=20),
            ModelCheckpoint(
        filepath=path_save_model_cp,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)]
training_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
history = training_model.fit(
    X_train_text, y_train_text,
    batch_size=1024,
    epochs=200,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1,
    
)
acc,  val_acc  = history.history['accuracy'], history.history['val_accuracy']
loss, val_loss = history.history['loss'], history.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
path_save_plt = os.path.join("d:\\NyeMan\\KULIAH S2\\Semester 1\\KK - B\\last-project","plots",'VAL_ACC_tf_text_weight_cp_data_paper_5.png')
# plt.show()
plt.savefig(path_save_plt)
plt.clf()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
path_save_plt = os.path.join("d:\\NyeMan\\KULIAH S2\\Semester 1\\KK - B\\last-project","plots",'VAL_LOSS_tf_text_weight_cp_data_paper_5.png')
# plt.show()
plt.savefig(path_save_plt)
plt.clf()

from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, accuracy_score

def ConfusionMatrix(y_pred, y_test):
    # Compute and plot the Confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred)
    print(cf_matrix)
    categories  = ['Negative','Positive']
    group_names = ['True Neg','False Pos', 'False Neg','True Pos']
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]

    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_names,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)

    sns.heatmap(cf_matrix, annot = labels, cmap = 'Blues',fmt = '',
                xticklabels = categories, yticklabels = categories)

    plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
    plt.ylabel("Actual values"   , fontdict = {'size':14}, labelpad = 10)
    plt.title ("Confusion Matrix", fontdict = {'size':18}, pad = 20)
    path_save_pltp = os.path.join("d:\\NyeMan\\KULIAH S2\\Semester 1\\KK - B\\last-project","plots",'ConfusionMatrix_tf_text_weight_cp_data_paper_5.png')
    # plt.show()
    plt.savefig(path_save_pltp)
    plt.clf()
    
# Predicting on the Test dataset.
y_pred = training_model.predict(X_test_text)

# Converting prediction to reflect the sentiment predicted.
y_pred = np.where(y_pred>=0.5, 1, 0)

print("precision", precision_score(y_test_text,y_pred))
print("recall", recall_score(y_test_text,y_pred))
print("f1 score", f1_score(y_test_text,y_pred))
print("accuracy", accuracy_score(y_test_text,y_pred))
ConfusionMatrix(y_pred, y_test_text)
print(classification_report(y_test_text, y_pred))


# Saving the TF-Model.
path_save_model = os.path.join("d:\\NyeMan\\KULIAH S2\\Semester 1\\KK - B\\last-project","models",'tf_text_data_paper_5')
training_model.save(path_save_model)
path_save_model = os.path.join("d:\\NyeMan\\KULIAH S2\\Semester 1\\KK - B\\last-project","models",'tf_text_weight_data_paper_5')
training_model.save_weights(path_save_model)