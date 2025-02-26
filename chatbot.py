import random
import json
import pickle
import numpy as np
import os

import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
nltk.download('punkt')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Define paths
intents_path = r'intents.json'
words_path = r'words.pkl'
classes_path = r'classes.pkl'
model_path = r'chatbot_model.keras'

# Load data
try:
    # Load JSON intents file
    if not os.path.isfile(intents_path):
        raise FileNotFoundError(f"File {intents_path} not found.")
    with open(intents_path, 'r') as file:
        intents = json.load(file)

    # Load words and classes pickle files
    if not os.path.isfile(words_path) or not os.path.isfile(classes_path):
        raise FileNotFoundError("Pickle files for words or classes not found.")
    with open(words_path, 'rb') as file:
        words = pickle.load(file)
    with open(classes_path, 'rb') as file:
        classes = pickle.load(file)
    print(classes)

    # Load model
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found.")
    model = load_model(model_path)

except FileNotFoundError as fnf_error:
    print(f"Error loading files: {fnf_error}")
    exit()
except json.JSONDecodeError:
    print("Error: intents.json is not a valid JSON file.")
    exit()
except pickle.UnpicklingError:
    print("Error: One of the pickle files is corrupt or incompatible.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    exit()

# Function definitions
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        if w in words:
            bag[words.index(w)] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]
    return return_list

def get_response(intents_list, intents_json):
    if not intents_list:
        return "Sorry, I didn't understand that."
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "Sorry, I don't have a response for that."

# Chat loop
while True:
    message = input('You: ')
    if message.lower() in ['exit', 'quit', 'bye']:
        print("Goodbye!")
        break
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(f"Bot: {res}")
