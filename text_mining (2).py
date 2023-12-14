#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
import spacy


# In[2]:


from matplotlib.pyplot import imread
from wordcloud import WordCloud, STOPWORDS
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# Dataset
tweets=pd.read_csv('C:\\Users\\Kavya\\Desktop\\DS Assignments\\ML\\Text mining\\Elon_musk.csv',encoding='Latin-1')


# In[4]:


tweets


# In[5]:


tweets.drop(['Unnamed: 0'],inplace=True,axis=1)


# In[6]:


tweets


# In[7]:


#Text Preprocessing
tweets=[Text.strip() for Text in tweets.Text]


# In[8]:


# remove both the leading and the trailing characters
tweets=[Text for Text in tweets if Text] 


# In[9]:


# removes empty strings, because they are considered in Python as False
tweets[0:10]


# In[10]:


# Joining the list into one string/text
tweets_text=' '.join(tweets)


# In[11]:


tweets_text


# In[12]:


# remove Twitter username handles from a given twitter text. (Removes @usernames)
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer(strip_handles=True)


# In[13]:


tweets_tokens=tknzr.tokenize(tweets_text)
print(tweets_tokens)


# In[14]:


# Again Joining the list into one string/text
tweets_tokens_text=' '.join(tweets_tokens)


# In[15]:


tweets_tokens_text


# In[16]:


# Remove Punctuations 
no_punc_text=tweets_tokens_text.translate(str.maketrans('','',string.punctuation))


# In[17]:


no_punc_text


# In[18]:


# remove https or url within text
import re
no_url_text=re.sub(r'http\S+', '', no_punc_text)


# In[19]:


no_url_text


# In[20]:


from nltk.tokenize import word_tokenize
text_tokens=word_tokenize(no_url_text)


# In[21]:


print(text_tokens)


# In[22]:


# Tokenization
import nltk


# In[23]:


nltk.download('punkt')


# In[24]:


nltk.download('stopwords')


# In[25]:


len(text_tokens)


# In[26]:


# Remove Stopwords
from nltk.corpus import stopwords
my_stop_words=stopwords.words('english')


# In[27]:


sw_list = ['\x92','rt','ye','yeah','haha','Yes','U0001F923','I']
my_stop_words.extend(sw_list)


# In[28]:


no_stop_tokens=[word for word in text_tokens if not word in my_stop_words]
print(no_stop_tokens)


# In[29]:


# Normalize the data
lower_words=[Text.lower() for Text in no_stop_tokens]
print(lower_words[100:200])


# In[30]:


# Stemming 
from nltk.stem import PorterStemmer
ps=PorterStemmer()


# In[31]:


stemmed_tokens=[ps.stem(word) for word in lower_words]
print(stemmed_tokens[100:200])


# In[32]:


# Lemmatization
import spacy
print(spacy.util.is_package('en_core_web_sm'))


# In[33]:


import spacy


# In[34]:


print(spacy.__version__)


# In[35]:


import spacy

# Download the 'en_core_web_sm' model
spacy.cli.download("en_core_web_sm")

# Load the model
nlp = spacy.load("en_core_web_sm")


# In[36]:


nlp=spacy.load('en_core_web_sm')
doc=nlp(' '.join(lower_words))
print(doc)


# In[37]:


lemmas=[token.lemma_ for token in doc]
print(lemmas)


# In[38]:


clean_tweets=' '.join(lemmas)
clean_tweets


# In[39]:


#Feature Extaction
#1. Using CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
tweetscv=cv.fit_transform(lemmas)
print(cv.vocabulary_)


# In[40]:


import sklearn
print(sklearn.__version__)
feature_names = list(cv.vocabulary_.keys())


# In[41]:


print(feature_names[100:200])


# In[42]:


print(tweetscv.toarray()[100:200])


# In[43]:


print(tweetscv.toarray().shape)


# In[44]:


#2. CountVectorizer with N-grams (Bigrams & Trigrams)
cv_ngram_range = CountVectorizer(analyzer='word', ngram_range=(1, 3), max_features=100)
bow_matrix_ngram = cv_ngram_range.fit_transform(lemmas)


# In[45]:


# Retrieve feature names
feature_names = cv_ngram_range.get_feature_names_out()
print(feature_names)
print(bow_matrix_ngram.toarray())


# In[46]:


#3. TF-IDF Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tfidfv_ngram_max_features=TfidfVectorizer(norm='l2',analyzer='word',ngram_range=(1,3),max_features=500)
tfidf_matix_ngram=tfidfv_ngram_max_features.fit_transform(lemmas)
print(tfidfv_ngram_max_features.get_feature_names_out())
print(tfidf_matix_ngram.toarray())


# In[47]:


#Generate Word Cloud
# Define a function to plot word cloud
def plot_cloud(wordcloud):
    plt.figure(figsize=(40,30))
    plt.imshow(wordcloud)
    plt.axis('off')


# In[48]:


# Generate Word Cloud
STOPWORDS.add('pron')
STOPWORDS.add('rt')
STOPWORDS.add('yeah')
wordcloud=WordCloud(width=3000,height=2000,background_color='black',max_words=50,
                   colormap='Set1',stopwords=STOPWORDS).generate(clean_tweets)
plot_cloud(wordcloud)


# In[49]:


#Named Entity Recognition (NER)
# Parts Of Speech (POS) Tagging
nlp=spacy.load('en_core_web_sm')


# In[50]:


one_block=clean_tweets
doc_block=nlp(one_block)
spacy.displacy.render(doc_block,style='ent',jupyter=True)


# In[51]:


for token in doc_block[100:200]:
    print(token,token.pos_) 


# In[52]:


# Filtering the nouns and verbs only
nouns_verbs=[token.text for token in doc_block if token.pos_ in ('NOUN','VERB')]
print(nouns_verbs[100:200])


# In[53]:


# Counting the noun & verb tokens
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()


# In[54]:


X=cv.fit_transform(nouns_verbs)
sum_words=X.sum(axis=0)


# In[55]:


words_freq=[(word,sum_words[0,idx]) for word,idx in cv.vocabulary_.items()]
words_freq=sorted(words_freq, key=lambda x: x[1], reverse=True)


# In[56]:


wd_df=pd.DataFrame(words_freq)
wd_df.columns=['word','count']
wd_df[0:10] # viewing top ten results


# In[57]:


# Visualizing results (Barchart for top 10 nouns + verbs)
wd_df[0:10].plot.bar(x='word',figsize=(12,8),title='Top 10 nouns and verbs');


# # Emotion Mining - Sentiment Analysis

# In[58]:


from nltk import tokenize
sentences=tokenize.sent_tokenize(' '.join(tweets))
sentences


# In[83]:


sent_df=pd.DataFrame(sentences,columns=['sentence'])
sent_df


# In[87]:


# Emotion Lexicon - Affin csv file
affin = pd.read_csv("C:\\data science\\afinn.csv",sep=',', encoding='Latin-1')


# In[88]:


affin


# In[89]:


affinity_scores = affin.set_index('word')['value'].to_dict()


# In[90]:


affinity_scores


# In[91]:


# Custom function: score each word in a sentence in lemmatised form, but calculate the score for the whole original sentence
nlp=spacy.load('en_core_web_sm')
sentiment_lexicon=affinity_scores

def calculate_sentiment(text:str=None):
    sent_score=0
    if text:
        sentence=nlp(text)
        for word in sentence:
            sent_score+=sentiment_lexicon.get(word.lemma_,0)
    return sent_score


# In[92]:


# manual testing
calculate_sentiment(text='great')


# In[93]:


# Calculating sentiment value for each sentence
sent_df['sentiment_value']=sent_df['sentence'].apply(calculate_sentiment)
sent_df['sentiment_value']


# In[94]:


# how many words are there in a sentence?
sent_df['word_count']=sent_df['sentence'].str.split().apply(len)
sent_df['word_count']


# In[95]:


sent_df.sort_values(by='sentiment_value')


# In[96]:


# Sentiment score of the whole review
sent_df['sentiment_value'].describe()


# In[97]:


# negative sentiment score of the whole review
sent_df[sent_df['sentiment_value']<=0]


# In[98]:


# positive sentiment score of the whole review
sent_df[sent_df['sentiment_value']>0]


# In[99]:


# Adding index cloumn
sent_df['index']=range(0,len(sent_df))
sent_df


# In[100]:


# Plotting the sentiment value for whole review
import seaborn as sns
plt.figure(figsize=(15,10))
sns.distplot(sent_df['sentiment_value'])


# In[101]:


# Plotting the line plot for sentiment value of whole review
plt.figure(figsize=(15,10))
sns.lineplot(y='sentiment_value',x='index',data=sent_df)


# In[102]:


# Correlation analysis
sent_df.plot.scatter(x='word_count',y='sentiment_value',figsize=(8,8),title='Sentence sentiment value to sentence word count')


# In[ ]:





# In[ ]:





# In[60]:


# 1) Extract reviews of any product from ecommerce website like amazon
# 2) Perform emotion mining


# In[61]:


import pandas as pd


# In[62]:


df = pd.read_csv("C:\\Users\\Kavya\\Desktop\\DS Assignments\\ML\\Text mining\\amazon_reviews.csv")


# In[63]:


df


# In[64]:


df = df.drop(["asin","title","location_and_date","verified"],axis=1)


# In[65]:


# SentimentIntensityAnalyzer
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()


# In[66]:


import string
# Remove punctuation from the text
translator = str.maketrans("", "", string.punctuation)
df['text'] = df['text'].apply(lambda text: text.translate(translator))


# In[67]:


df["text"]


# In[68]:


# Perform sentiment analysis on each text and create a new column for sentiment scores
df['sentiment_score'] = df['text'].apply(lambda text: sia.polarity_scores(text))


# In[69]:


df['sentiment_score'] 


# In[70]:


df


# In[71]:


# Assume d1["sentiment_score"] contains the dictionary as mentioned above
df['neg_score'] = df['sentiment_score'].apply(lambda x: x['neg'])
df['neu_score'] = df['sentiment_score'].apply(lambda x: x['neu'])
df['pos_score'] = df['sentiment_score'].apply(lambda x: x['pos'])
df['compound_score'] = df['sentiment_score'].apply(lambda x: x['compound'])


# In[72]:


df


# In[73]:


# Classify sentiments into categories (positive, negative, neutral)
df['sentiment_label'] = df['sentiment_score'].apply(lambda scores: 'positive' if scores['compound'] > 0 else 'negative' if scores['compound'] < 0 else 'neutral')


# In[74]:


df


# In[75]:


# Split the data for training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['sentiment_label'], test_size=0.3, random_state=42)


# In[76]:


# Vectorize the text data
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(stop_words='english')
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)


# In[77]:


# Initialize the Logistic Regression model
from sklearn.linear_model import LogisticRegression
LE = LogisticRegression()


# In[78]:


LE.fit(X_train_vect, y_train)


# In[79]:


# Predictions on training set
y_train_pred = LE.predict(X_train_vect)


# In[80]:


# Predictions on test set
y_test_pred = LE.predict(X_test_vect)


# In[81]:


# Accuracy
from sklearn.metrics import accuracy_score
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)


# In[82]:


print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)


# In[ ]:




