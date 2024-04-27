# naan-mudhalvan
i
    MaxPooling1D(5),
    Flatten(),
    Dense(1, activation='sigmoid')
])

# Compile CNN model
cnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train CNN model
cnn_model.fit(x_train, y_train, batch_size=128, epochs=5, validation_split=0.2)

# Define ANN model
ann_model = Sequential([
    Embedding(input_dim=num_words, output_dim=128, input_length=maxlen),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile ANN model
ann_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train ANN model
ann_model.fit(x_train, y_train, batch_size=128, epochs=5, validation_split=0.2)

# Evaluate models on test data
rnn_test_loss, rnn_test_accuracy = rnn_model.evaluate(x_test, y_test)
cnn_test_loss, cnn_test_accuracy = cnn_model.evaluate(x_test, y_test)
ann_test_loss, ann_test_accuracy = ann_model.evaluate(x_test, y_test)

# Calculate total accuracy
total_accuracy = (rnn_test_accuracy + cnn_test_accuracy + ann_test_accuracy) / 3

print(f"RNN Test Accuracy: {rnn_test_accuracy}")
print(f"CNN Test Accuracy: {cnn_test_accuracy}")
print(f"ANN Test Accuracy: {ann_test_accuracy}")
print(f"Total Accuracy: {total_accuracy}")
mport numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load IMDB Movie Review dataset
num_words = 10000  # Consider only the top 10,000 words in the dataset
maxlen = 200  # Maximum length of a review (truncate or pad to this length)
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)

# Pad sequences to ensure uniform length
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

# Define RNN model
rnn_model = Sequential([
    Embedding(input_dim=num_words, output_dim=128, input_length=maxlen),
    LSTM(64),
    Dense(1, activation='sigmoid')
])

# Compile RNN model
rnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train RNN model
rnn_model.fit(x_train, y_train, batch_size=128, epochs=5, validation_split=0.2)

# Define CNN model
cnn_model = Sequential([
    Embedding(input_dim=num_words, output_dim=128, input_length=maxlen),
    Conv1D(32, 5, activation='relu'),
