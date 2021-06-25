''' Load libraries '''
import pandas as pd                                             # For Data Exploration, 
import numpy as np                                              # To create arrays
import nltk                                                     # For Text Pre-processing 
import re                                                       # For Text Pre-processing
from nltk.tokenize import word_tokenize                         # Tokenize text into words
from nltk.stem import PorterStemmer                             # Reducing word to it's root
from sklearn.feature_extraction.text import CountVectorizer     # Create Bag of Words
from sklearn.model_selection import train_test_split            # Split data into groups (Testing and Training)

from nltk.corpus import stopwords
from string import punctuation
trashwords = stopwords.words('english')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import load_model

import pickle

# Use in case you get an error trying to import stopwords
# nltk.download('stopwords') 

def load_data():
    '''
    Import your data into the program and display it
    
    Task: Load dataset and display dataset
    Hint: Using pandas will make your life a lot easier
    '''
    df = pd.read_csv('emails.csv', usecols = ["text", "spam"])
    print(df)
    
    '''
    Check for any Null Values (empty rows) and drop duplicate rows
    
    Task: Eliminate empty and duplicate rows
    Hint: Use pandas!
    '''
    df.drop_duplicates(inplace = True)
    df.dropna(inplace = True)
    
    
    '''
    Now it's time to start cleaning. Let's remove any unnecessary pieces of text.
    
    Hint: Display one piece of text to see what we should remove
    Task: Iterate over rows and perform cleaning, then display your dataset again
    '''
    # print(df['text'][0])
    
    for index,row in df.iterrows():
        new_text = re.sub('Subject: |re : |fw : |fwd : ', '', row['text'])
        new_text = new_text.lower().strip()
        df.loc[index,'text'] = new_text
    
    # print(df)
        
    '''
    Create your final corpus of sentences. The corpus must be a list of all sentences
    in its stemmed form and should not include punctuation characters or stopwords.
    
    Task: Create a list of strings containing each stemmed and processed sentence.
    Hint: Tokenize each sentence to handle words separately. Use word_tokenize to
    tokenize and PortStemmer() to stem.
    '''
    corpus = []
    stemmer = PorterStemmer()
    for text in df['text']:
        tokenized_text = word_tokenize(text)
        stemmed_text = ''
        for word in tokenized_text:
            if word not in punctuation and word not in trashwords:
                stemmed_text += stemmer.stem(word) + ' '
        corpus.append(stemmed_text)
        
    
    '''
    Create a Bag of Words representation of your corpus (x) and a list of the
    labels (y). Both must have the same length!
    
    Task: Create a Bag of Words model and its respective list of labels
    Hint: Use scikit's CountVectorizer()
    '''
    cv = CountVectorizer()
    x = cv.fit_transform(corpus).toarray()
    y = df.iloc[:,1].values
    
    
    '''
    Save the data structures to your system.
    '''
    # Use pickle to save the data structures
    pickle.dump(x, open('x.pkl', 'wb'))
    pickle.dump(y, open('y.pkl', 'wb'))
    pickle.dump(cv, open('cv.pkl', 'wb'))
    
    return x, y, cv


def create_network(x_train, y_train):
    '''
    Load the model saved in system, or else create one using the given input and 
    output training sets. 
    
    Task: Create a Neural Network with x_train as input and y_train as output
    Hint: Use tensorflow's Sequential() for the NN and Dense() for each layer.
    '''
    try:
        model = load_model('model.h5')
    except:
        model = Sequential()
        model.add(Dense(800, input_shape = (len(x_train[0]),), activation="relu"))
        model.add(Dense(400, activation="relu"))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(loss="binary_crossentropy", metrics = ["accuracy"])
        
        model.fit(x_train, y_train, epochs=8, batch_size = 100)
        
        model.save('model.h5')
        
    return model


def main():
    '''
    Open the data structures and store them to respective variables.
    '''
    # Use pickle to load the data structures
    try:
        x = pickle.load(open('x.pkl', 'rb'))
        y = pickle.load(open('y.pkl', 'rb'))
        cv = pickle.load(open('cv.pkl', 'rb'))
    except:
        x, y, cv = load_data()
        
    
    '''
    Split your data into a training set and a testing set. We chose 20% for the
    test size, but you can tweak this value and see how it affects the final result.
    '''
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
    
    
    '''
    Load the Neural Network model
    '''
    model = create_network(x_train, y_train)

    
    '''
    Evaluate model with testing sets.
    '''
    results = model.evaluate(x_test, y_test)
    print('Accuracy: {:.2%}'.format(results[1]))
    
    
    '''
    Evaluate model with new user's input.
    '''
    user_text = input('Input the text: ')
    prediction = model.predict(cv.transform([user_text]))[0]
    print('Spam level: {:.2%}'.format(prediction[0]))
    if prediction > 0.8:
        print('Spam!')
    else:
        print('Not spam!')

main()
