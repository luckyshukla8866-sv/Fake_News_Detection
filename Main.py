import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

data = pd.read_csv("Fake_News_Detection/Data/news.csv")
print(data.head())

data = data.drop(["Unnamed: 0"], axis=1)
print(data.head(5))

#Variable Setup
le = preprocessing.LabelEncoder()
le.fit(data['label'])
data['label'] = le.transform(data['label'])

#Variables Setup 
embedding_dim = 50
max_length = 54
padding_type = 'post'
trunc_type = 'post'
oov_tok = "<OOV>"
training_size = 3000
test_portion = 0.1

# Tokenization 
title = []
text = []
labels = []
for x in range(training_size):
    title.append(data['title'][x])
    text.append(data['text'][x])
    labels.append(data['label'][x])

tokenizer1 = Tokenizer()
tokenizer1.fit_on_texts(title)
word_index1 = tokenizer1.word_index
vocab_size1 = len(word_index1)
sequences1 = tokenizer1.texts_to_sequences(title)
padded1 = pad_sequences(sequences1, padding=padding_type, truncating=trunc_type)

#Splitting Data
split = int(test_portion * training_size)
training_sequences1 = padded1[split:training_size]
test_sequences1 = padded1[0:split]
test_labels = labels[0:split]
training_labels = labels[split:training_size]

#Reshaping Data for LSTM
training_sequences1 = np.array(training_sequences1)
test_sequences1 = np.array(test_sequences1)

#Generating Word Embedding
embedding_index = {}
with open('Fake_News_Detection/glove.6B/glove.6B.50d.txt', 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs
        
embedding_matrix = np.zeros((vocab_size1 + 1, embedding_dim))

for word, i in word_index1.items():
    if i < vocab_size1:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

#Model Architecture
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size1 + 1, embedding_dim, input_length=max_length, 
                              weights=[embedding_matrix], trainable=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv1D(64, 5, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=4),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

#Training the Model
history = model.fit(
    training_sequences1, 
    np.array(training_labels), 
    epochs=50, 
    validation_data=(test_sequences1, np.array(test_labels)), 
    verbose=2
)

#Sample Prediction
X = "Karry to go to France in gesture of sympathy"

sequences = tokenizer1.texts_to_sequences([X])
sequences = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
if model.predict(sequences, verbose=0)[0][0] >= 0.5:
    print("This news is True")
else:
    print("This news is False")

    