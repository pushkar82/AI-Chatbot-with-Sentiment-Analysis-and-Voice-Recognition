import random
import json
import pickle
import numpy as np
import os
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

class Chatbot:
    def __init__(self, model_path="chatbot_model.keras", intents_path="intents.json",
                 words_path="words.pkl", classes_path="classes.pkl", max_length=150):
        # Initialize lemmatizer
        nltk.download('punkt')
        self.lemmatizer = WordNetLemmatizer()
        self.max_length = max_length

        # Load data
        try:
            with open(intents_path, 'r') as file:
                self.intents = json.load(file)
            with open(words_path, 'rb') as file:
                self.words = pickle.load(file)
            with open(classes_path, 'rb') as file:
                self.classes = pickle.load(file)
            self.model = load_model(model_path)
        except Exception as e:
            raise FileNotFoundError(f"Error loading files: {e}")

    def clean_up_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        return [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]

    def bag_of_words(self, sentence):
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(self.words)
        for w in sentence_words:
            if w in self.words:
                bag[self.words.index(w)] = 1
        return np.array(bag)

    def predict_class(self, sentence):
        bow = self.bag_of_words(sentence)
        res = self.model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return [{'intent': self.classes[r[0]], 'probability': str(r[1])} for r in results]

    def get_response(self, intents_list):
        if not intents_list:
            return "Sorry, I didn't understand that."
        tag = intents_list[0]['intent']
        for i in self.intents['intents']:
            if i['tag'] == tag:
                return random.choice(i['responses'])
        return "Sorry, I don't have a response for that."

    def generate_response(self, prompt):
        intents_list = self.predict_class(prompt)
        return self.get_response(intents_list)
