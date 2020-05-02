import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

text = input("Write a bit of news: ")
print("Text is:",text)
print("Loading Model...")
pkl_filename = "model.pkl"
try:
    with open(pkl_filename, 'rb') as file:
        model = pickle.load(file)
        tf1 = pickle.load(open("tfidf1.pkl", 'rb'))
    print("Model Loaded.")
except:
    print("Error: Model not found")
    


# Create new tfidfVectorizer with old vocabulary
tf1_new = TfidfVectorizer(analyzer='word', stop_words = "english", vocabulary = tf1.vocabulary_)

X_temp = tf1_new.fit_transform([text])
X_temp.toarray()
predict = model.predict(X_temp)
print(predict)