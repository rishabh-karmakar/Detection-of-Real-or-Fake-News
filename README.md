# Detection-of-Real-or-Fake-News

# PROBLEM STATEMENT

To build a model to accurately classify a piece of news as REAL or FAKE.

Using sklearn,  build a **TfidfVectorizer** on the provided dataset. Then, initialize a **PassiveAggressive Classifier** and fit the model. In the end, the accuracy score and the confusion matrix tell us how well our model fares.

## Description of all files:
* news.zip: Unzip the Dataset to get news.csv
* news.csv: Dataset having fake and real news
* Real_or_Fake_News.ipynb: Jupyter Notebook containing all explanation and my workdoings
* train.py: Simply run this file to automatically train the model and generate vocabulary and model.pkl file to be saved for further <br><b>Could be run only once</b>
<br>*Takes a command line argument taking the file name*
<br>*Usage* 
```bash{cmd=True}
python train.py news.csv
```
* predict.py: Run this file as much as you want. Uses the saved models to run, hence is much faster to execute.
<br>P.S The larger the text, the better the chance of accurate prediction
```
python predict.py
```
![Annotation of cmd](https://user-images.githubusercontent.com/48029688/80854614-2b9d6480-8c57-11ea-8669-7bade98a2aa3.png)

### Disclamer: I donot guarrantee in case some real life sensitive words comes off as fake. It is solely trained on the dataset. No feelings are meant to be hurt.
