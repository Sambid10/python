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

lemmatizer=WordNetLemmatizer()
file_path="D:/py/current/intents.json"

with open(file_path, 'r') as file:
        intents = json.load(file)
words=[]
classes=[]
documents=[]
ignore_symbols=['?','!','.',':']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list=nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list,intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words=[lemmatizer.lemmatize(word) for word in words if word not in ignore_symbols]

words=sorted(set(words))
classes=sorted(set(classes))

pickle.dump(words,open('D:/py/current/model/words.pkl','wb'))

pickle.dump(classes,open('D:/py/current/model/classes.pkl','wb'))

training=[]
output_empty=[0] * len(classes)

for document in documents:
    bag=[]
    words_pattern=document[0]
    words_pattern=[lemmatizer.lemmatize(word) for word in words_pattern if word not in ignore_symbols]

    for word in words:
        bag.append(1) if word in   words_pattern else bag.append(0)
    
    output_ro=list(output_empty)
    output_ro[classes.index(document[1])]=1

    training.append([bag,output_ro])


random.shuffle(training)
training=np.array(training,dtype=object)

train_x=list(training[:,0])
train_y=list(training[:,1])

model=Sequential()
model.add(Dense(128,input_shape=(len(train_x[0]),),activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]),activation='softmax'))

sgd=SGD(nesterov=True,momentum=0.9,learning_rate=0.01,)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

model.fit(np.array(train_x),np.array(train_y),epochs=200,batch_size=5,verbose=1)
model.save('D:/py/current/model/chatbot_model.keras')
