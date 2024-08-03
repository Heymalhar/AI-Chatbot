import numpy as np
import json
import random
import pickle
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Load the intents file
intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignoreLetters = ['!', '?', '.', ',']

# Iterate over each intent in the intents file
for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignoreLetters]
words = sorted(set(words))

# Sort classes
classes = sorted(set(classes))

# Save words and classes
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Create the training data
training = []
outputEmpty = [0] * len(classes)

for document in documents:
    bag = []
    wordPatterns = document[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]

    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)

    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)

# Shuffle and convert to numpy array
random.shuffle(training)
training = np.array(training)

# Create training and testing lists
trainX = training[:, :len(words)]
trainY = training[:, len(words):]

# Create the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))

# Compile the model
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
hist = model.fit(np.array(trainX), np.array(trainY), epochs=200, batch_size=5, verbose=True)

# Save the model
model.save('chatbot_model.h5', hist)

print("The model has been trained and saved as 'chatbot_model.h5'.")
