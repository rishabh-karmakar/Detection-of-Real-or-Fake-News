import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


try:
    if(len(sys.argv)==1):
        print("Please enter Filename: ")
    else:
        print("File name is: ",sys.argv[1])
        news = pd.read_csv(sys.argv[1])
        print("File Loaded")
        print("The dimentions are: ",news.shape)
        print('There are: ',news['label'].value_counts(),"values")

        # Adding a label as 0 for fake and 1 for real
        news['label_num'] = news['label'].map({'FAKE': 0, 'REAL': 1})

        # Another column with all texts including title and text
        news['data'] = news['title'] + ' ' + news['text']

        X = news['data']
        Y = news['label']
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1705)

        print("Creating the TFIDF Vectorizer")
        tfidf_vectorizer = TfidfVectorizer(analyzer='word', stop_words="english")
        tfidf_train = tfidf_vectorizer.fit_transform(x_train)
        tfidf_test = tfidf_vectorizer.transform(x_test)

        pickle.dump(tfidf_vectorizer, open("tfidf1.pkl", "wb"))

        print("Initializing Passive Aggressive Classifier ")
        # Initialize a PassiveAggressiveClassifier
        pac = PassiveAggressiveClassifier(max_iter=100)
        pac.fit(tfidf_train, y_train)

        y_pred = pac.predict(tfidf_test)

        Score = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {round(Score * 100, 2)}%')

        # Build confusion matrix
        print("Confusion Matrix is:",confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL']))

        #Saving the model
        pkl_filename = "model.pkl"
        with open(pkl_filename, 'wb') as file:
            pickle.dump(pac, file)

        print("Training and Saving model complete")
except:
    print("Error")