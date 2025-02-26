import os
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

#For logging level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

lemmatizer = WordNetLemmatizer()

intents = json.load(open(r'intents.json'))

# Preparing lists for processing
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# Process intents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and remove duplicates
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))

# Sort classes
classes = sorted(set(classes))

# Save words and classes
pickle.dump(words, open(r'words.pkl', 'wb'))
pickle.dump(classes, open(r'classes.pkl', 'wb'))

# Prepare training data
training = []
output_empty = [0] * len(classes)
print(documents)
for document in documents:
    bag = [1 if word in document[0] else 0 for word in words]
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# Shuffle and convert training data to NumPy array

random.shuffle(training)
training = np.array(training, dtype=object)

# Split into features and labels
train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# Build model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu')) #input layer
model.add(Dropout(0.5))
model.add(Dense(128,input_shape=(len(train_x[0]),), activation='relu')) #hidden layer
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax')) # output layer

# Compile model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train model
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Save model
import os

path = r'chatbot_model.keras'
if os.path.exists(path):
    print("Path exists.")
else:
    print("Path does not exist.")

model.save(r'chatbot_model.keras')
