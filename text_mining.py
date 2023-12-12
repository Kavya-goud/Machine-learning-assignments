#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
import spacy


# In[3]:


from matplotlib.pyplot import imread
from wordcloud import WordCloud, STOPWORDS
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


# Dataset
tweets=pd.read_csv('C:\\Users\\Kavya\\Desktop\\DS Assignments\\ML\\Text mining\\Elon_musk.csv',encoding='Latin-1')


# In[5]:


tweets


# In[6]:


tweets.drop(['Unnamed: 0'],inplace=True,axis=1)


# In[7]:


tweets


# In[8]:


#Text Preprocessing
tweets=[Text.strip() for Text in tweets.Text]


# In[9]:


# remove both the leading and the trailing characters
tweets=[Text for Text in tweets if Text] 


# In[10]:


# removes empty strings, because they are considered in Python as False
tweets[0:10]


# In[11]:


# Joining the list into one string/text
tweets_text=' '.join(tweets)


# In[12]:


tweets_text


# In[13]:


# remove Twitter username handles from a given twitter text. (Removes @usernames)
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer(strip_handles=True)


# In[14]:


tweets_tokens=tknzr.tokenize(tweets_text)
print(tweets_tokens)


# In[15]:


# Again Joining the list into one string/text
tweets_tokens_text=' '.join(tweets_tokens)


# In[29]:


tweets_tokens_text


# In[17]:


# Remove Punctuations 
no_punc_text=tweets_tokens_text.translate(str.maketrans('','',string.punctuation))


# In[18]:


no_punc_text


# In[19]:


# remove https or url within text
import re
no_url_text=re.sub(r'http\S+', '', no_punc_text)


# In[20]:


no_url_text


# In[21]:


from nltk.tokenize import word_tokenize
text_tokens=word_tokenize(no_url_text)


# In[22]:


print(text_tokens)


# In[23]:


# Tokenization
import nltk


# In[24]:


nltk.download('punkt')


# In[25]:


nltk.download('stopwords')


# In[26]:


len(text_tokens)


# In[27]:


# Remove Stopwords
from nltk.corpus import stopwords
my_stop_words=stopwords.words('english')


# In[30]:


sw_list = ['\x92','rt','ye','yeah','haha','Yes','U0001F923','I']
my_stop_words.extend(sw_list)


# In[31]:


no_stop_tokens=[word for word in text_tokens if not word in my_stop_words]
print(no_stop_tokens)


# In[32]:


# Normalize the data
lower_words=[Text.lower() for Text in no_stop_tokens]
print(lower_words[100:200])


# In[33]:


# Stemming 
from nltk.stem import PorterStemmer
ps=PorterStemmer()


# In[34]:


stemmed_tokens=[ps.stem(word) for word in lower_words]
print(stemmed_tokens[100:200])


# In[37]:


# Lemmatization
import spacy
print(spacy.util.is_package('en_core_web_sm'))


# In[38]:


import spacy


# In[41]:


print(spacy.__version__)


# In[43]:


import spacy

# Download the 'en_core_web_sm' model
spacy.cli.download("en_core_web_sm")

# Load the model
nlp = spacy.load("en_core_web_sm")


# In[45]:


nlp=spacy.load('en_core_web_sm')
doc=nlp(' '.join(lower_words))
print(doc)


# In[46]:


lemmas=[token.lemma_ for token in doc]
print(lemmas)


# In[47]:


clean_tweets=' '.join(lemmas)
clean_tweets


# In[48]:


#Feature Extaction
#1. Using CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
tweetscv=cv.fit_transform(lemmas)
print(cv.vocabulary_)


# In[50]:


import sklearn
print(sklearn.__version__)
feature_names = list(cv.vocabulary_.keys())


# In[51]:


print(feature_names[100:200])


# In[52]:


print(tweetscv.toarray()[100:200])


# In[53]:


print(tweetscv.toarray().shape)


# In[54]:


#2. CountVectorizer with N-grams (Bigrams & Trigrams)
cv_ngram_range = CountVectorizer(analyzer='word', ngram_range=(1, 3), max_features=100)
bow_matrix_ngram = cv_ngram_range.fit_transform(lemmas)


# In[55]:


# Retrieve feature names
feature_names = cv_ngram_range.get_feature_names_out()
print(feature_names)
print(bow_matrix_ngram.toarray())


# In[56]:


#3. TF-IDF Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tfidfv_ngram_max_features=TfidfVectorizer(norm='l2',analyzer='word',ngram_range=(1,3),max_features=500)
tfidf_matix_ngram=tfidfv_ngram_max_features.fit_transform(lemmas)
print(tfidfv_ngram_max_features.get_feature_names_out())
print(tfidf_matix_ngram.toarray())


# In[57]:


#Generate Word Cloud
# Define a function to plot word cloud
def plot_cloud(wordcloud):
    plt.figure(figsize=(40,30))
    plt.imshow(wordcloud)
    plt.axis('off')


# In[58]:


# Generate Word Cloud
STOPWORDS.add('pron')
STOPWORDS.add('rt')
STOPWORDS.add('yeah')
wordcloud=WordCloud(width=3000,height=2000,background_color='black',max_words=50,
                   colormap='Set1',stopwords=STOPWORDS).generate(clean_tweets)
plot_cloud(wordcloud)


# In[59]:


#Named Entity Recognition (NER)
# Parts Of Speech (POS) Tagging
nlp=spacy.load('en_core_web_sm')


# In[60]:


one_block=clean_tweets
doc_block=nlp(one_block)
spacy.displacy.render(doc_block,style='ent',jupyter=True)


# In[61]:


for token in doc_block[100:200]:
    print(token,token.pos_) 


# In[62]:


# Filtering the nouns and verbs only
nouns_verbs=[token.text for token in doc_block if token.pos_ in ('NOUN','VERB')]
print(nouns_verbs[100:200])


# In[63]:


# Counting the noun & verb tokens
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()


# In[64]:


X=cv.fit_transform(nouns_verbs)
sum_words=X.sum(axis=0)


# In[65]:


words_freq=[(word,sum_words[0,idx]) for word,idx in cv.vocabulary_.items()]
words_freq=sorted(words_freq, key=lambda x: x[1], reverse=True)


# In[66]:


wd_df=pd.DataFrame(words_freq)
wd_df.columns=['word','count']
wd_df[0:10] # viewing top ten results


# In[67]:


# Visualizing results (Barchart for top 10 nouns + verbs)
wd_df[0:10].plot.bar(x='word',figsize=(12,8),title='Top 10 nouns and verbs');


# In[68]:


from nltk import tokenize
sentences=tokenize.sent_tokenize(' '.join(tweets))
sentences


# In[69]:


sent_df=pd.DataFrame(sentences,columns=['sentence'])
sent_df


# In[ ]:





# In[ ]:





# In[33]:


# 1) Extract reviews of any product from ecommerce website like amazon
# 2) Perform emotion mining


# In[70]:


import pandas as pd


# In[71]:


df = pd.read_csv("C:\\Users\\Kavya\\Desktop\\DS Assignments\\ML\\Text mining\\amazon_reviews.csv")


# In[72]:


df


# In[73]:


df = df.drop(["asin","title","location_and_date","verified"],axis=1)


# In[74]:


# SentimentIntensityAnalyzer
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()


# In[75]:


import string
# Remove punctuation from the text
translator = str.maketrans("", "", string.punctuation)
df['text'] = df['text'].apply(lambda text: text.translate(translator))


# In[76]:


df["text"]


# In[77]:


# Perform sentiment analysis on each text and create a new column for sentiment scores
df['sentiment_score'] = df['text'].apply(lambda text: sia.polarity_scores(text))


# In[78]:


df['sentiment_score'] 


# In[79]:


df


# In[80]:


# Assume d1["sentiment_score"] contains the dictionary as mentioned above
df['neg_score'] = df['sentiment_score'].apply(lambda x: x['neg'])
df['neu_score'] = df['sentiment_score'].apply(lambda x: x['neu'])
df['pos_score'] = df['sentiment_score'].apply(lambda x: x['pos'])
df['compound_score'] = df['sentiment_score'].apply(lambda x: x['compound'])


# In[81]:


df


# In[82]:


# Classify sentiments into categories (positive, negative, neutral)
df['sentiment_label'] = df['sentiment_score'].apply(lambda scores: 'positive' if scores['compound'] > 0 else 'negative' if scores['compound'] < 0 else 'neutral')


# In[83]:


df


# In[84]:


# Split the data for training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['sentiment_label'], test_size=0.3, random_state=42)


# In[85]:


# Vectorize the text data
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(stop_words='english')
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)


# In[86]:


# Initialize the Logistic Regression model
from sklearn.linear_model import LogisticRegression
LE = LogisticRegression()


# In[87]:


LE.fit(X_train_vect, y_train)


# In[88]:


# Predictions on training set
y_train_pred = LE.predict(X_train_vect)


# In[89]:


# Predictions on test set
y_test_pred = LE.predict(X_test_vect)


# In[90]:


# Accuracy
from sklearn.metrics import accuracy_score
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)


# In[91]:


print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)


# In[ ]:




