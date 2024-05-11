
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Hyperparameters
max_features = 10000  # Maximum number of words to consider
max_len = 200  # Maximum review length

# Load the IMDB dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# Pad sequences to have the same length
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

# Define the model
model = Sequential()
model.add(Embedding(max_features, 128, input_length=max_len))  # Word embedding layer
model.add(LSTM(64))  # LSTM layer
model.add(Dense(1, activation='sigmoid'))  # Output layer with sigmoid activation

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

# Evaluate the model on test data
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print("Test Accuracy:", accuracy)

# Predict sentiment for a new review
new_review = "This movie is fantastic!"  # Preprocess and pad the review similarly to training data
predicted_sentiment = model.predict(np.array([new_review]))[0][0]

if predicted_sentiment > 0.5:
  print("Predicted sentiment: Positive")
else:
  print("Predicted sentiment: Negative")
