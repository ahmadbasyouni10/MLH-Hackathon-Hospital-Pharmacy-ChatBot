import nltk
#used to stem the word 
from nltk.stem.lancaster import LancasterStemmer 
stemmer = LancasterStemmer()

import numpy as np
import tflearn
import random 
import tensorflow as tf
import random 
import json
#data for the model, if used before we don't need to run the model code
import pickle 

#stores json file data into python dictionary
with open("intents.json") as file:
    data = json.load(file)

#rb read bytes (data is saved as bytes)
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    #initialization
    words = []
    labels = []
    docs_x = []
    docs_y = []

    #loop through every dictionary in intents
    #splits all the patterns into individual words and adds them to list using tokenize
    #creates a list with all patterns that has a value from another list with its tag
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            individual = nltk.word_tokenize(pattern)
            words.extend(individual)
            docs_x.append(individual)
            docs_y.append(intent["tag"])
        #adds tags that are not in label
        if intent["tag"] not in labels:
            labels.append(intent["tag"])
        
    #ensures all words are in lower case and takes their stem to 
    #get rid of duplicates using set then make words a list again, then sort them
    #makes sure we ignore question marks
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    #hot encoded for neural network use makes a empty list from 0 to length of labels
    out_empty = [0 for _ in range(len(labels))]

    #provides index and value using enumerate
    for x, doc in enumerate(docs_x):
        bag = []

        #stems the individual list which has individual words
        individual = [stemmer.stem(w) for w in doc]

        #loops and adds the word into the bag if in the list individual
        for w in words:
            if w in individual:
                bag.append(1)
            else:
                bag.append(0)
        
        #inserts a one in output row where tag is found in docs y of corresponding x
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        #list of 0s and 1s for training and output
        training.append(bag)
        output.append(output_row)

    #transform the lists into numpy arrays
    training = np.array(training)
    output = np.array(output)
    
    #write bytes
    #write words, labels, training, and output into a pickle file
    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output),f)

#model
tf.compat.v1.reset_default_graph()

#softmax gives us probability for every neuron in layer = output
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

#dont train model if we already did
try:
    model.load("model.tflearn")
except:
    #fitting the model
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)

    model.save("model.tflearn")

#analyzes user input sentence
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for sen in s_words:
        for i, w in enumerate(words):
            if w ==sen:
                bag[i]=(1)
    
    return np.array(bag)

#chatbot usage
#makes prediction on model
#finds index in results with highest probability
#stores label at that index into tag
#prints a random response from the data imported from json if it equals tag found from labels that has all the tags(at index)
def chat():
    print("Start talking with the bot! (Type quit to exit)")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = np.argmax(results)
        tag = labels[results_index]

        if results[results_index] > 0.6:
            for tg in data["intents"]:
                #from data imported if the tag in that data equals tag initialized from labels, print response from data
                if tg["tag"] == tag:
                    responses = tg["responses"]
            
            print(random.choice(responses))
        
        else:
            print("I didnt understand, try a different question!")



chat()