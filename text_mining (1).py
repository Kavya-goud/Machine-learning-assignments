#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import string


# In[2]:


df = pd.read_csv("C:\\Users\\Kavya\\Desktop\\DS Assignments\\ML\\Text mining\\Elon_musk.csv " , encoding = 'ISO-8859-1')


# In[3]:


df


# In[4]:


# Dropping a column
data = df.iloc[:,0:1]


# In[5]:


data


# In[6]:


# stopwords
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
stop_words


# In[7]:


len(stop_words)


# In[8]:


stop_words = list(stop_words)


# In[9]:


stop_words


# In[10]:


# load positive word
with open("C:\\Users\\Kavya\\Desktop\\DS Assignments\\ML\\Text mining\\positive-words.txt", "r") as file:
    positive_words = file.read()


# In[11]:


# Extract only the lines with actual words (exclude lines starting with ';')
positive_words_list = [line.strip() for line in positive_words.split('\n') if not line.startswith(';') and line.strip()]


# In[12]:


positive_words_list


# In[13]:


# load negative words
with open("C:\\Users\\Kavya\\Desktop\\DS Assignments\\ML\\Text mining\\negative-words.txt", "r") as file:
    negative_words = file.read()


# In[14]:


# Extract only the lines with actual words (exclude lines starting with ';')
negative_words_list = [line.strip() for line in negative_words.split('\n') if not line.startswith(';') and line.strip()]


# In[15]:


negative_words_list


# In[16]:


# Initialize the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()


# In[17]:


# Remove punctuation from the text
translator = str.maketrans("", "", string.punctuation)
df['Text'] = df['Text'].apply(lambda text: text.translate(translator))


# In[18]:


# Perform sentiment analysis on each tweet and create a new column for sentiment scores
df['Sentiment_Score'] = df['Text'].apply(lambda text: sia.polarity_scores(text)['compound'])


# In[19]:


df['Sentiment_Score']


# In[20]:


df


# In[21]:


# Classify sentiments into categories (positive, negative, neutral)
df['Sentiment_Label'] = df['Sentiment_Score'].apply(lambda score: 'positive' if score > 0 else 'negative' if score < 0 else 'neutral')


# In[22]:


df


# In[23]:


x = df['Text']


# In[24]:


x


# In[25]:


y = df['Sentiment_Label']


# In[26]:


y


# In[27]:


vectorizer = CountVectorizer(stop_words=stopwords)


# In[28]:


# STEMMING
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
for i in df['Text'].index:
    df['Text'].iloc[i] = stemmer.stem(df['Text'].iloc[i])
df['Text']


# In[29]:


df


# In[30]:


#TOKENIZATION 
from sklearn.feature_extraction.text import CountVectorizer
Vectorizer = CountVectorizer()
Vt = Vectorizer.fit_transform(df['Text'])
Vt.toarray()


# In[31]:


from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer().fit(Vt)
x_vect = transformer.transform(Vt)


# In[32]:


x_vect


# In[ ]:





# In[33]:


# 1) Extract reviews of any product from ecommerce website like amazon
# 2) Perform emotion mining


# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv("C:\\Users\\Kavya\\Desktop\\DS Assignments\\ML\\Text mining\\amazon_reviews.csv")


# In[3]:


df


# In[5]:


df = df.drop(["asin","title","location_and_date","verified"],axis=1)


# In[6]:


# SentimentIntensityAnalyzer
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()


# In[8]:


import string
# Remove punctuation from the text
translator = str.maketrans("", "", string.punctuation)
df['text'] = df['text'].apply(lambda text: text.translate(translator))


# In[9]:


df["text"]


# In[11]:


# Perform sentiment analysis on each text and create a new column for sentiment scores
df['sentiment_score'] = df['text'].apply(lambda text: sia.polarity_scores(text))


# In[12]:


df['sentiment_score'] 


# In[13]:


df


# In[15]:


# Assume d1["sentiment_score"] contains the dictionary as mentioned above
df['neg_score'] = df['sentiment_score'].apply(lambda x: x['neg'])
df['neu_score'] = df['sentiment_score'].apply(lambda x: x['neu'])
df['pos_score'] = df['sentiment_score'].apply(lambda x: x['pos'])
df['compound_score'] = df['sentiment_score'].apply(lambda x: x['compound'])


# In[16]:


df


# In[18]:


# Classify sentiments into categories (positive, negative, neutral)
df['sentiment_label'] = df['sentiment_score'].apply(lambda scores: 'positive' if scores['compound'] > 0 else 'negative' if scores['compound'] < 0 else 'neutral')


# In[19]:


df


# In[20]:


# Split the data for training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['sentiment_label'], test_size=0.3, random_state=42)


# In[21]:


# Vectorize the text data
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(stop_words='english')
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)


# In[23]:


# Initialize the Logistic Regression model
from sklearn.linear_model import LogisticRegression
LE = LogisticRegression()


# In[24]:


LE.fit(X_train_vect, y_train)


# In[26]:


# Predictions on training set
y_train_pred = LE.predict(X_train_vect)


# In[27]:


# Predictions on test set
y_test_pred = LE.predict(X_test_vect)


# In[28]:


# Accuracy
from sklearn.metrics import accuracy_score
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)


# In[29]:


print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)


# In[ ]:




