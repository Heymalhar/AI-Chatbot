from flask import Flask, render_template, request, jsonify

import random
import pickle
import json
import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer
from keras.models import load_model

app = Flask(__name__, static_folder='static')

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]

    ERROR_THRESHOLD = 0.25

    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x:x[1], reverse=True)

    return_list = []

    for r in results:
        return_list.append({'intent':classes[r[0]], 'probability':str(r[1])})

    return return_list

def get_response(intents_list, intents_json):
    list_of_intents = intents_json['intents']
    tag = intents_list[0]['intent']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    
    return result

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/chat', methods=['POST', 'GET'])
def chat():
    message = request.json['message']
    ints = predict_class(message)
    res = get_response(ints, intents)
    return jsonify({'response': res})

if __name__ == "__main__":
    print("The chabot is good to go!")
    print("Go to Port 8000 to strike a conversation >>>>")
    print("")
    app.run(debug=True, port=8000)