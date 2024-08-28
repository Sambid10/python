import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import random
import pickle
import json

import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer

from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense,Activation,Dropout
from keras._tf_keras.keras.optimizers import SGD
nltk.download('punkt_tab')
nltk.download('wordnet')

from keras._tf_keras.keras.models import load_model

def clean_up_sentence(sentence):
    lemmatizer= WordNetLemmatizer()
    ignore_symbols=["?","!",".",","]
    sentence=nltk.word_tokenize(sentence)
    sentence=[lemmatizer.lemmatize(word) for word in sentence if word not in ignore_symbols]
    return sentence


def bag_of_words(sentence):
    words=pickle.load(open("D:/py/current/model/words.pkl",'rb'))

    sentence=clean_up_sentence(sentence)
    bag=[0]*len(words) 
    
    for w in sentence:
        for i, word in enumerate(words):
            if  word == w:
                bag[i]=1
    return np.array(bag)

def predict_class(sentence):
    classes=pickle.load(open("D:/py/current/model/classes.pkl",'rb'))
    model=load_model("D:/py/current/model/chatbot_model.keras")

    bag=bag_of_words(sentence)
    res=model.predict(np.array([bag]))[0]
    ERROR_THRESHOLD=0.25
    results=[[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x:x[1],reverse=True)

    return_list=[]
    for r in results:
        return_list.append({'intent':classes[r[0]],'probability':str(r[1])})
    
    return return_list

def get_response(intents_list):
    intents_json=json.load(open("D:/py/current/intents.json"))
    tag=intents_list[0]['intent']
    list_of_intents=intents_json['intents']
    result = "Sorry, I don't understand."

    for i in list_of_intents:
        if i['tag'] == tag:
            result=random.choice(i['responses'])
            break
    
    return result
