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

#IMPORT DATASET
path_this_file = os.path.dirname(os.path.abspath(__file__))
file_data = "fully-labeled-dataset.csv"
cols=['id','tweet_text','text','emojis','sentiment_label','emoji_label','sarcasm_label']
path_data = os.path.join("d:\\NyeMan\\KULIAH S2\\Semester 1\\KK - B\\last-project","data",file_data)
df = pd.read_csv(path_data,sep="|")
df = df[['text','emojis','sentiment_label','emoji_label','sarcasm_label']]
print(df.head())

ax = df.copy()[['text','sentiment_label']].groupby('sentiment_label').count().plot(kind='bar', title='Distribution of Sentiment Label',
                                               legend=True)
ax = ax.set_xticklabels(['Negative','Positive'], rotation=0)
# plt.show()

ax = df.copy()[['emojis','emoji_label']].groupby('emoji_label').count().plot(kind='bar', title='Distribution of Emoji Label',
                                               legend=True)
ax = ax.set_xticklabels(['Negative','Positive','Neutral'], rotation=0)
# plt.show()
ax = df.copy()[['text','emojis','sarcasm_label']].groupby('sarcasm_label').count().plot(kind='bar', title='Distribution of Sarcasm Label',
                                               legend=True)
ax = ax.set_xticklabels(['Negative','Positive'], rotation=0)
plt.show()

#PREPROCESSING
#skipped

#SPLITTING THE DATA
from sklearn.model_selection import train_test_split
X_data, y_data = np.array(df[['text','emojis']]), np.array(df[['sentiment_label','emoji_label','sarcasm_label']])

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data,
                                                    test_size = 0.25, random_state = 50, stratify= (y_data[:,[0]]).flatten())
print(X_train, y_train)
X_train_text = (X_train[:,[0]]).flatten()
X_train_emoji = (X_train[:,[1]]).flatten()
X_test_text = (X_test[:,[0]]).flatten()
X_test_emoji = (X_test[:,[1]]).flatten()

y_train_sentiment = (y_train[:,[0]]).flatten()
y_train_emoji = (y_train[:,[1]]).flatten()
y_train_sarcasm = (y_train[:,[2]]).flatten()

y_test_sentiment = (y_test[:,[0]]).flatten()
y_test_emoji = (y_test[:,[1]]).flatten()
y_test_sarcasm = (y_test[:,[2]]).flatten()

# for i in range(0, len(X_train_text)):
#     print(X_train_text[i], y_train_sentiment[i])

# for i in range(0, len(X_test_text)):
#     print(X_test_text[i], y_test_sentiment[i])

# WORD EMBEDDING
from gensim.models import Word2Vec

Embedding_dimensions = 100

# Creating Word2Vec training dataset.
Word2vec_train_data_text = list(map(lambda x: x.split(), X_train_text))
Word2vec_train_data_emoji = list(map(lambda x: x.split(), X_train_emoji))


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
word2vec_model_emoji = Word2Vec(Word2vec_train_data_emoji,
                 vector_size=20,
                 workers=8,
                 window=5,
                 alpha=0.05,
                 epochs=30,
                 min_count=2)
path_save_model = os.path.join("d:\\NyeMan\\KULIAH S2\\Semester 1\\KK - B\\last-project","models",'word2vec_model_emoji.model')
word2vec_model_emoji.save(path_save_model)
print("Vocabulary Length:", len(word2vec_model_emoji.wv.key_to_index))

# Defining the model input length.
input_length = 100

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
vocab_length = 2400

tokenizer_text = Tokenizer(filters="", lower=False, oov_token="<oov>")
tokenizer_text.fit_on_texts((X_data[:,[0]]).flatten())
tokenizer_text.num_words = vocab_length
print("Tokenizer Text vocab length:", vocab_length)

tokenizer_emoji = Tokenizer(filters="", lower=False, oov_token="<oov>")
tokenizer_emoji.fit_on_texts((X_data[:,[1]]).flatten())
tokenizer_emoji.num_words = 500
print("Tokenizer Emoji vocab length:", vocab_length)

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
from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D, Dense, LSTM, Conv1D, Embedding

def getModel():
    embedding_layer = Embedding(input_dim = vocab_length,
                                output_dim = Embedding_dimensions,
                                weights=[embedding_matrix],
                                input_length=input_length,
                                trainable=True)

    model = Sequential([
        embedding_layer,
        Bidirectional(LSTM(50, dropout=0.3, return_sequences=True)),
        Bidirectional(LSTM(32, dropout=0.3, return_sequences=True)),
        Bidirectional(LSTM(50, dropout=0.3, return_sequences=True)),
        Conv1D(2, 5, activation='relu'),
        GlobalMaxPool1D(),
        Dense(10, activation='relu'),
        Dense(1, activation='sigmoid'),
    ],
    name="Sentiment_Model")
    return model

training_model = getModel()
training_model.summary()
print(X_train_text.shape, y_train_sentiment.shape)
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

path_save_model_cp = os.path.join("d:\\NyeMan\\KULIAH S2\\Semester 1\\KK - B\\last-project","models",'tf_text_weight_cp_lsl_paper')

callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),
            EarlyStopping(monitor='val_accuracy', min_delta=1e-4, patience=5),
            ModelCheckpoint(
        filepath=path_save_model_cp,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)]
training_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
history = training_model.fit(
    X_train_text, y_train_sentiment,
    batch_size=1024,
    epochs=50,
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

plt.show()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

from sklearn.metrics import confusion_matrix, classification_report

def ConfusionMatrix(y_pred, y_test):
    # Compute and plot the Confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred)

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
    plt.show()
# Predicting on the Test dataset.
y_pred = training_model.predict(X_test_text)

# Converting prediction to reflect the sentiment predicted.
y_pred = np.where(y_pred>=0.5, 1, 0)

# Printing out the Evaluation metrics. 
ConfusionMatrix(y_pred, y_test_sentiment)
# Print the evaluation metrics for the dataset.
print(classification_report(y_test_sentiment, y_pred))


# Saving the TF-Model.
path_save_model = os.path.join("d:\\NyeMan\\KULIAH S2\\Semester 1\\KK - B\\last-project","models",'tf_text_lsl_paper')
training_model.save(path_save_model)
path_save_model = os.path.join("d:\\NyeMan\\KULIAH S2\\Semester 1\\KK - B\\last-project","models",'tf_text_weight_lsl_paper')
training_model.save_weights(path_save_model)