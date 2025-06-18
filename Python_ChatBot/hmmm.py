from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# 🟢 This route serves your chatbot frontend
@app.route('/')
def home():
    return render_template('index.html')  # This must exist in a folder named 'templates'

# 🟢 This handles chatbot POST requests
@app.route('/get', methods=['POST'])
def chatbot_response():
    user_msg = request.form['msg']
    
    import random
    import json
    import pickle
    import numpy as np
    import nltk

    from nltk.stem import WordNetLemmatizer
    from keras.models import load_model

    nltk.download('punkt')  # Only needed once

    lemmatizer = WordNetLemmatizer()
    intents = json.loads(open('intents.json').read())

    words = pickle.load(open('word.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
    model = load_model('chat_bot_model.h5')

    def clean_up_sentence(sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [lemmatizer.lemmatize(word, pos='n') for word in sentence_words]  # ✅ Safe lemmatizing
        return sentence_words

    def bag_of_words(sentence):
        sentence_words = clean_up_sentence(sentence)
        bag = [0] * len(words)
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
        results.sort(key=lambda x: x[1], reverse=True)
        return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]

    def get_response(intents_list, intents_json):
        if not intents_list:
            return "I'm sorry, I didn't understand that."
    
    tag = intents_list[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])

    print("GO! Bot is running!")

    while True:
        message = input("")
        ints = predict_class(message)
        res = get_response(ints, intents)
        print(res)
        return jsonify({"response": user_msg})

if __name__ == '__main__':
    app.run(debug=True)
    