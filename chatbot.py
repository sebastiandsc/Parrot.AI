import nltk
import random
import string
import warnings

warnings.filterwarnings('ignore')

textFile = open("Datasets/1.txt", 'r', errors="ignore")

rawText = textFile.read()
rawText = rawText.lower()

sentences = nltk.sent_tokenize(rawText) # This uses nltk to separate the SENTENCES in the text.

words = nltk.word_tokenize(rawText) #This uses nltk to separate the WORDS in the text.


# Fix up the text to be used.

simplifier = nltk.stem.WordNetLemmatizer()

def SimplifiedWords(inputs):
    return [simplifier.lemmatize(input) for input in inputs]


removePunctuation = dict((ord(punctuation), None) for punctuation in string.punctuation)

def Simplify(text):
    return SimplifiedWords(nltk.word_tokenize(text.lower().translate(removePunctuation)))


# List of greetings

GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey", "how's it going", "good day", "how are you?", "yo")
GREETING_RESPONSES = ["hi", "hey", "Yo!", "Hi there!", "Hello!", "Greetings!", "What's up?", "How's it going?", "Thank you for using Parrot.AI! How can I help?"]

def greeting(sentence):

    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# Vectorizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



def response(userInput):
    chatbotResponse = ''
    sentences.append(userInput)
    TfidfVector = TfidfVectorizer(tokenizer=Simplify, stop_words="english") # We separate the word bank into TFIDF vectors. 
    tfidfMatrix = TfidfVector.fit_transform(sentences) # We convert the Vectors to a Matrix
    Values = cosine_similarity(tfidfMatrix[-1], tfidfMatrix) # We compute the cosine similarity and store it.
    index = Values.argsort()[0][-2] # Calculate the index 
    flattenedArray = Values.flatten() # Flatten the array obtained through cosine similarity
    flattenedArray.sort()
    requiredTfidf = flattenedArray[-2] # This is the required TFidf
    if(requiredTfidf == 0): # If it is zero, this means the word is not present in the bot's vocabulary. 
        chatbotResponse = "Woops! Sorry, I guess I didn't understand you there."
        return chatbotResponse
    else:
        chatbotResponse = chatbotResponse + sentences[index]
        return chatbotResponse
"""
if __name__ == "__main__":
    print("BOT: Hello! Thank you for using Parrot.AI! Ask me anything!")
    while(True):
        userInput = input()
        userInput = userInput.lower()
        if(userInput=="bye"):
            print("BOT: Bye bye!")
            break
        else:
            if("thank" in userInput):
                print("BOT: You're welcome!")
                break
            else:
                if(greeting(userInput)!=None):
                    print("BOT: " + greeting(userInput))
                else:
                    string = "BOT: "
                    string = string + response(userInput)
                    print(string)
                    sentences.remove(userInput)
"""

def MessageFunction(msg):
    userInput = msg.lower()
    if(userInput=="bye"):
        string = "Bye bye!"
        return string
    else:
        if("thank" in userInput):
            string = "You're welcome!"
            return string
        else:
            if(greeting(userInput)!=None):
                string = greeting(userInput)
                return string
            else:
                string = response(userInput)
                sentences.remove(userInput)
                return string

#Flask
from flask import Flask, render_template, request, jsonify, make_response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
	return render_template("home.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    msgRes = str(MessageFunction(userText))
    print(msgRes)
    return jsonify(answer=msgRes)

if __name__ == "__main__":
	app.run()

                
    


    

