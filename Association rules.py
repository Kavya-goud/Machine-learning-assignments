#!/usr/bin/env python
# coding: utf-8

# # QUESTION - 1(BOOK CSV FILE)

# In[14]:


from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import pandas as pd


# In[15]:


df = pd.read_csv("C:\\Users\\Kavya\\Desktop\\DS Assignments\\ML\\Association rules\\book.csv")


# In[16]:


df


# In[17]:


# Define a range of support and confidence values
support_values = [0.1, 0.2, 0.3, 0.4, 0.5]
confidence_values = [0.5, 0.6, 0.7, 0.8, 0.9]


# In[18]:


# Iterate through support and confidence values
for support in support_values:
    for confidence in confidence_values:
        # Run Apriori algorithm
        frequent_itemsets = apriori(df, min_support=support, use_colnames=True)
        # Generate association rules
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=confidence)
        # Print or store results
        print(f"Support={support}, Confidence={confidence}, Number of Rules={len(rules)}")


# In[19]:


# Define a range of minimum length values
min_length_values = [1, 2, 3]


# In[20]:


# Iterate through minimum length values
for min_length in min_length_values:
    # Run Apriori algorithm with the specified minimum length
    frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)
    
    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    
    # Filter rules based on minimum length
    filtered_rules = rules[rules['antecedents'].apply(len) >= min_length]
    
    # Print or store results
    print(f"Minimum Length={min_length}, Number of Rules={len(filtered_rules)}")


# In[21]:


#Visualize the obtained rules using different plots
import matplotlib.pyplot as plt
import seaborn as sns


# In[22]:


# Scatter plot of Support vs. Confidence
plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Support vs. Confidence Scatter Plot')
plt.show()


# In[23]:


# Histogram of Rule Distribution based on Support
sns.histplot(rules['support'], bins=20, kde=True)
plt.xlabel('Support')
plt.ylabel('Number of Rules')
plt.title('Rule Distribution based on Support')
plt.show()


# In[24]:


# Network graph
import networkx as nx


# In[25]:


G = nx.Graph()
for i in range(len(rules)):
    G.add_node(str(rules.iloc[i]['antecedents']))
    G.add_node(str(rules.iloc[i]['consequents']))
    G.add_edge(str(rules.iloc[i]['antecedents']), str(rules.iloc[i]['consequents']))


# In[26]:


nx.draw(G, with_labels=True)
plt.title('Association Rules Network Graph')
plt.show()


# In[18]:


pip install wordcloud


# In[27]:


# Word cloud of frequent itemsets
from wordcloud import WordCloud


# In[28]:


wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df.columns))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Itemset Word Cloud')
plt.show()


# # QUESTION - 2(MY_MOVIES CSV FILE)

# In[1]:


from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import pandas as pd


# In[2]:


df = pd.read_csv("C:\\Users\\Kavya\\Desktop\\DS Assignments\\ML\\Association rules\\my_movies.csv")


# In[3]:


df


# In[4]:


df_new = df.iloc[:,5:]


# In[5]:


df_new


# In[6]:


# Define a range of support and confidence values
support_values = [0.1, 0.2, 0.3, 0.4, 0.5]
confidence_values = [0.5, 0.6, 0.7, 0.8, 0.9]


# In[7]:


# Iterate through support and confidence values
for support in support_values:
    for confidence in confidence_values:
        # Run Apriori algorithm
        frequent_itemsets = apriori(df_new, min_support=support, use_colnames=False)
        
        # Generate association rules
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=confidence)
        
        # Print or store results
        print(f"Support={support}, Confidence={confidence}, Number of Rules={len(rules)}")


# In[9]:


# Define a range of minimum length values
min_length_values = [1, 2, 3]


# In[10]:


# Iterate through minimum length values
for min_length in min_length_values:
    # Run Apriori algorithm with the specified minimum length
    frequent_itemsets = apriori(df_new, min_support=0.2, use_colnames=True)
    
    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    
    # Filter rules based on minimum length
    filtered_rules = rules[rules['antecedents'].apply(len) >= min_length]
    
    # Print or store results
    print(f"Minimum Length={min_length}, Number of Rules={len(filtered_rules)}")


# In[11]:


#Visualize the obtained rules using different plots.
import matplotlib.pyplot as plt
import seaborn as sns


# In[12]:


# Scatter plot of Support vs. Confidence
plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Support vs. Confidence Scatter Plot')
plt.show()


# In[13]:


# Histogram of Rule Distribution based on Support
sns.histplot(rules['support'], bins=20, kde=True)
plt.xlabel('Support')
plt.ylabel('Number of Rules')
plt.title('Rule Distribution based on Support')
plt.show()


# In[14]:


# Network graph
import networkx as nx


# In[15]:


G = nx.Graph()
for i in range(len(rules)):
    G.add_node(str(rules.iloc[i]['antecedents']))
    G.add_node(str(rules.iloc[i]['consequents']))
    G.add_edge(str(rules.iloc[i]['antecedents']), str(rules.iloc[i]['consequents']))


# In[16]:


nx.draw(G, with_labels=True)
plt.title('Association Rules Network Graph')
plt.show()


# In[17]:


# Word cloud of frequent itemsets
from wordcloud import WordCloud


# In[18]:


wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df_new.columns))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Itemset Word Cloud')
plt.show()


# In[ ]:




