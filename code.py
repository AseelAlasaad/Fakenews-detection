import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
import re
import string 

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


real_data = pd.read_csv('/fake-and-real-news-dataset/True.csv')
fake_data = pd.read_csv('/fake-and-real-news-dataset/Fake.csv')

real_data.head()

fake_data.head()
#add column 
real_data['label'] = 1
fake_data['label'] = 0 
#Merging the 2 datasets
data = pd.concat([real_data, fake_data], ignore_index=True, sort=False)
data.head()

# checking distribution of data 

plt.figure(figsize = (9, 5))
sns.countplot(data['label'])

plt.show()


#Visualization
print(data["target"].value_counts())
fig, ax = plt.subplots(1,2, figsize=(19, 5))
g1 = sns.countplot(data.target,ax=ax[0],palette="pastel");
g1.set_title("Count of real and fake data")
g1.set_ylabel("Count")
g1.set_xlabel("Target")
g2 = plt.pie(data["target"].value_counts().values,explode=[0,0],labels=data.target.value_counts().index, autopct='%1.1f%%',colors=['SkyBlue','PeachPuff'])
fig.show()

# data cleaning
data['text']= data['subject'] + " " + data['title'] + " " + data['text']
del data['title']
del data['subject']
del data['date']
data.head()

first_text = data.text[10]
first_text

#remove Punctuation Marks and Special Characters
first_text = re.sub('\[[^]]*\]', ' ', first_text)
first_text = re.sub('[^a-zA-Z]',' ',first_text)  # replaces non-alphabets with spaces
first_text = first_text.lower() # Converting from uppercase to lowercase
first_text

# remove stop word 

nltk.download("stopwords")   
from nltk.corpus import stopwords  

# we can use tokenizer instead of split
first_text = nltk.word_tokenize(first_text)

first_text = [ word for word in first_text if not word in set(stopwords.words("english"))]


#Lemmatization: transform words into their base form
lemma = nltk.WordNetLemmatizer()
first_text = [ lemma.lemmatize(word) for word in first_text] 

first_text = " ".join(first_text)



#Removal of Punctuation Marks
def remove_punctuations(text):
    return re.sub('\[[^]]*\]', '', text)

# Removal of Special Characters
def remove_characters(text):
    return re.sub("[^a-zA-Z]"," ",text)

#Removal of stopwords 
def remove_stopwords_and_lemmatization(text):
    final_text = []
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    for word in text:
        if word not in set(stopwords.words('english')):
            lemma = nltk.WordNetLemmatizer()
            word = lemma.lemmatize(word) 
            final_text.append(word)
    return " ".join(final_text)

#Total function
def cleaning(text):
    text = remove_punctuations(text)
    text = remove_characters(text)
    text = remove_stopwords_and_lemmatization(text)
    return text

#Apply function on text column
data['text']=data['text'].apply(cleaning)




# splitting dataset into train data and test data


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

transformer = TfidfTransformer(smooth_idf=False)
count_vectorizer = CountVectorizer(ngram_range=(1, 2))
counts = count_vectorizer.fit_transform(data['text'].values)
X = transformer.fit_transform(counts)

#separating the data and label
X = data['text'].values
Y = data['label'].values

X_train, X_test,Y_train, Y_test = train_test_split(X, Y, test_size = 0.20)

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, RocCurveDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier




model = LogisticRegression()
model.fit(X_train, Y_train)
model_train_score = model.score(X_train, Y_train)
model_test_score = model.score(X_test, Y_test)
Y_pred = model.predict(X_test)
print(classification_report(Y_test, Y_pred))
cm = confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(8,5))
sns.heatmap(cm, annot=True, fmt='.2f');



from sklearn.svm import SVC
model = SVC()
model.fit(X_train, Y_train)
model_train_score = model.score(X_train, Y_train)
model_test_score = model.score(X_test, Y_test)
Y_pred = model.predict(X_test)
print(classification_report(Y_test, Y_pred))
cm = confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(8,5))
sns.heatmap(cm, annot=True, fmt='.2f');



