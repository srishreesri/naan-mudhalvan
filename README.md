
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

max_features = 10000  
max_len = 200  # Maximum review length

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

model = Sequential()
model.add(Embedding(max_features, 128, input_length=max_len))  # Word embedding layer
model.add(LSTM(64))  # LSTM layer
model.add(Dense(1, activation='sigmoid'))  # Output layer with sigmoid activation

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print("Test Accuracy:", accuracy)

new_review = "This movie is fantastic!"  # Preprocess and pad the review similarly to training data
predicted_sentiment = model.predict(np.array([new_review]))[0][0]

if predicted_sentiment > 0.5:
  print("Predicted sentiment: Positive")
else:
  print("Predicted sentiment: Negative")
