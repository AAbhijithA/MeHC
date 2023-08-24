'''
Importing necessary libraries for preprecessing/training the model
'''
import pickle
import random
import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers.legacy import SGD

'''
Loading the intents.json() dataset and other dependencies
'''
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words = []
classes = []
documents = []
ignore = ['?','!','.',',']
nltk.download('punkt')
nltk.download('wordnet')

'''
Pre-Processing
'''
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list,intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore]
words = sorted(set(words))
classes = sorted(set(classes))

'''
Training the Model: Sequential NN with Stochastic Gradient Descent Optimizer
'''
training = []
output_empty = [0]*len(classes)
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        if word in word_patterns:
            bag.append(1)
        else:
            bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag,output_row])
random.shuffle(training)

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model_save = model.fit(np.array(train_x), np.array(train_y), epochs=300, batch_size=10, verbose=1)

'''
Saving the: model we trained and the pre-processed classes and words for later use
'''
model.save('mhcb.h5',model_save)
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))
