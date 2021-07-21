from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

# Misc libraries
import pandas as pd
import nltk
import os
import pickle

PREPROCESSED_DATA_PATH = "data/cleaned_news_dataset.json"
PREPROCESSED_TEST_DATA_PATH = "data/cleaned_post_data.csv"

# Function to load data into pandas dataframe
def load_data(path: str):
    if path.endswith(".json"):
        return pd.read_json(path,lines=True)
    else:
        return pd.read_csv(path)       

def check_for_cleaned_train_data(path: str):
    # Load cleaned data from file if exists to save time
    if os.path.isfile(PREPROCESSED_DATA_PATH):
        df = pd.read_json(PREPROCESSED_DATA_PATH, lines=True)
    else:
        # Preprocess data and save to file
        df = load_data(path)
        features = df['headline']
        labels = df['category']
        features = preprocess_data(features)
        df = pd.concat([features,labels], axis=1)
        df.to_json(PREPROCESSED_DATA_PATH, orient='records', lines=True)
    
    return df

def check_for_cleaned_test_data(path: str):
    # Load cleaned data from file if exists to save time
    if os.path.isfile(PREPROCESSED_TEST_DATA_PATH):
        df = pd.read_csv(PREPROCESSED_TEST_DATA_PATH)
    else:
        # Preprocess data and save to file
        df = load_data(path)

        # Remove invalid rows
        df['Title transcription'].replace('', float("NaN"), inplace=True)
        df.dropna(subset=['Title transcription'], inplace=True)
        df = df[df['Title transcription'] != 'No Title']
        df = df[df['Title transcription'] != 'No title']
        df = df[df['Title transcription'] != 'No title ']
        df = df[df['Title transcription'] != 'No Title.']
        df = df[df['Title transcription'] != 'No title.']
        df = df[df['Title transcription'] != 'no title']
        df = df[df['Title transcription'] != 'None']
        df = df[df['Title transcription'] != 'None.']
        df = df[df['Title transcription'] != 'none']
        df = df[df['Title transcription'] != 'none ']

        # Return preprecessed data in Pandas dataframe
        features = df['Title transcription']
        df = preprocess_data(features)
        df.to_csv(PREPROCESSED_TEST_DATA_PATH)
    
    return df

# Function to preprocess/clean text data
def preprocess_data(df: pd.DataFrame):
    # Sets all words to lowercase and removes punctuation
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    df = df.apply(lambda sentence: [w for w in tokenizer.tokenize(sentence.lower())])

    # Removes "stop words" and digits which do not provide additional meaning to sentence
    stop_words = set(nltk.corpus.stopwords.words('english'))
    df = df.apply(lambda sentence: [w for w in sentence if w not in stop_words and not w.isdigit()])

    # Lemmatizes words (reduces them to their root/base word)
    lemmatizer = nltk.stem.WordNetLemmatizer()
    df = df.apply(lambda sentence: [lemmatizer.lemmatize(w) for w in sentence])

    # Finally changes list back into string to be used by later algorithms
    df = df.apply(lambda sentence: ' '.join(sentence))

    return df

# Function to run predictions on the test data
def test_model(data_path: str, model_path: str):
    # load in the classifier and test data
    clf, vectorizer, encoder = pickle.load(open(model_path, 'rb'))
    df = check_for_cleaned_test_data(data_path)

    # Vectorize the test features
    features = vectorizer.transform(df['Title transcription'])
    
    # Run predictions on those features
    preds = clf.predict(features)
    preds = pd.concat([df,pd.Series(encoder.inverse_transform(preds))], axis=1).rename(columns={"Unnamed: 0": "Index", "Title transcription": "processed title", 0: "Predicted Category"}).set_index('Index')

    # Join the predictions with the original data
    original_data = pd.read_csv(data_path)
    joined_data = original_data.join(preds, how='outer').iloc[:,1:]

    # Save the predicitons to a csv
    joined_data.to_csv('predictions/'+model_path.split('/')[1]+'.csv')
    print("Made Predictions Here: predictions/"+model_path.split('/')[1]+'.csv')

# Function to train MultinomialNB model
def train_mnb(path: str, filename: str):
    df = check_for_cleaned_train_data(path) 

    vectorizer = CountVectorizer()
    features = vectorizer.fit_transform(df['headline'])

    encoder = LabelEncoder()
    labels = encoder.fit_transform(df['category'])

    nb = MultinomialNB()
    print("5-fold cross validation average training accuracy: ",cross_val_score(nb, features, labels, cv=5).mean())
    
    nb.fit(features, labels)
    pickle.dump((nb, vectorizer, encoder), open('models/'+filename,'wb'))
    print("Saved model: models/"+filename)

# Function to train Stochastic Gradient Descent model
def train_sgd(path: str, filename: str):
    df = check_for_cleaned_train_data(path) 

    vectorizer = CountVectorizer()
    features = vectorizer.fit_transform(df['headline'])

    encoder = LabelEncoder()
    labels = encoder.fit_transform(df['category'])

    # Converts data to Tfidf format
    # Max iter is 15 instead of 1000 in order to save time
    tfidf = TfidfTransformer()
    sgd = SGDClassifier(max_iter=15)

    pipeline = Pipeline(steps=[('tfidf', tfidf),('sgd',sgd)])
    parameters = {
        'sgd__penalty': ['l1', 'l2'],
        'sgd__alpha': [1e-3, 1e-4]
    }

    # Trains a search grid of parameters to find best fit
    # CV = 3 in an effort to spend less time training
    clf = GridSearchCV(pipeline, parameters,cv=3)

    predictions = cross_val_predict(clf,features,labels,cv=3)

    print("3-fold cross validation training accuracy: ", accuracy_score(labels,predictions))

    clf.fit(features, labels)
    pickle.dump((clf, vectorizer, encoder), open('models/'+filename,'wb'))
    print("Saved model: models/"+filename)

# Function to train simple neural network
def train_nn(path: str, filename: str):
    df = check_for_cleaned_train_data(path) 

    vectorizer = CountVectorizer()
    features = vectorizer.fit_transform(df['headline'])

    encoder = LabelEncoder()
    labels = encoder.fit_transform(df['category'])

    # Converts data to Tfidf format
    # Max iter is 100 instead of 1000 in order to save time
    tfidf = TfidfTransformer()
    mlp = MLPClassifier(solver='adam', hidden_layer_sizes=(30), max_iter=100)

    pipeline = Pipeline(steps=[('tfidf', tfidf),('mlp',mlp)])
    parameters = {
        'mlp__hidden_layer_sizes': [(30,), (40,),(50,)],
        'mlp__activation': ['logistic', 'relu']
    }

    # Trains a search grid of parameters to find best fit
    # CV = 3 in an effort to spend less time training
    clf = GridSearchCV(pipeline, parameters,cv=3)

    predictions = cross_val_predict(clf,features,labels,cv=3)

    print("3-fold cross validation training accuracy: ", accuracy_score(labels,predictions))

    clf.fit(features, labels)
    pickle.dump((clf, vectorizer, encoder), open('models/'+filename,'wb'))
    print("Saved model: models/"+filename)