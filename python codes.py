
# coding: utf-8

# In[2]:



from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import hstack
from sklearn.pipeline import make_union
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from nltk.corpus import stopwords
import csv
from textblob import TextBlob
import os
import pandas as pd
import string
import re
import textmining
import matplotlib.pyplot as plt
from wordcloud import wordcloud,STOPWORDS
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
stop_words = set(stopwords.words('english'))
from sklearn.metrics import accuracy_score
os.chdir("F:\P CODE")
df=pd.read_csv("train.csv")
df_new=df
df_2_test=pd.read_csv("test.csv")
subm=pd.read_csv('sample_submission.csv')


# In[3]:


from sklearn.metrics import log_loss,confusion_matrix,classification_report,roc_curve,auc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn import metrics


# In[108]:


################################################################################################################################


# In[109]:


precleaning visualisation
Number of comments
df_toxic=df.drop(['id','comment_text'],axis=1)
counts=[]
categories=list(df_toxic.columns.values)
for i in categories:
   counts.append((i,df_toxic[i].sum()))

df_stats=pd.DataFrame(counts,columns=['category','Number of comments'])
df_stats

df_stats.plot(x='category',y='Number of comments',kind='bar',legend='False',grid=True,figsize=(8,5))
plt.title("Number of comments per category")
plt.ylabel('Number of occurences',fontsize=12)
plt.xlabel('category',fontsize=12)


# In[ ]:


# Number of words
df['word_count'] = df['comment_text'].apply(lambda x: len(str(x).split(" ")))
df[['comment_text','word_count']].head()


# In[ ]:


# Number of characters
df['char_count'] = df['comment_text'].str.len() ## this also includes spaces
df[['comment_text','char_count']].head()


# In[ ]:


# Average word length:
def avg_word(sentence):
  words = sentence.split()
  return (sum(len(word) for word in words)/len(words))

df['avg_word'] = df['comment_text'].apply(lambda x: avg_word(x))
df[['comment_text','avg_word']].head()


# In[ ]:


# Number of stopwords
df['stopwords'] = df['comment_text'].apply(lambda x: len([x for x in x.split() if x in stop]))
df[['comment_text','stopwords']].head()


# In[ ]:


# Number of numeric
df['numerics'] = df['comment_text'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
df[['comment_text','numerics']].head()


# In[ ]:


# number of uppercase words
df['upper'] = df['comment_text'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
df[['comment_text','upper']].head()


# In[ ]:


###############################################################################################################################


# In[ ]:


# Data cleaning....................


# In[110]:


import HTMLParser


# In[111]:


# html_parser=HTMLParser.HTMLParser()


# In[125]:


# def clean_html(ht):
#     ht=html_parser.unescape(ht)
#     return str(ht)
# df['comment_text'] = df['comment_text'].map(lambda com : clean_html(com))
# df_2_test['comment_text'] = df_2_test['comment_text'].map(lambda com : clean_html(com))


# In[126]:


# df.comment_text[1504]=re.sub('https?://[A-za-z0-9./]+','',df.comment_text[1504])


# In[136]:


# lowercase convert
# df['comment_text'] = df['comment_text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
# df_2_test['comment_text'] = df_2_test['comment_text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
# df['comment_text'].head()


# In[4]:


# Data preprocessing

def clean_text(text):
    text = text.lower()
#     text = re.sub('https?://[A-za-z0-9./]+','',text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text


df_2_test['comment_text'] = df_2_test['comment_text'].map(lambda com : clean_text(com))
df['comment_text'] = df['comment_text'].map(lambda com : clean_text(com))
df['comment_text'].head(5)


# In[140]:


# cleaned_train_comment = []
# for i in range(0,len(df)):
#     cleaned_comment = clean_text(df['comment_text'][i])
#     cleaned_train_comment.append(cleaned_comment)
# df['comment_text'] = pd.Series(cleaned_train_comment).astype(str)



# In[141]:


# cleaned_test_comment = []
# for i in range(0,len(df_2_test)):
#     cleaned_comment_test = clean_text(df_2_test['comment_text'][i])
#     cleaned_test_comment.append(cleaned_comment_test)
# df_2_test['comment_text'] = pd.Series(cleaned_test_comment).astype(str)


# In[6]:


df_2_test.head(5)


# In[7]:


stop=set(stopwords.words('english'))
exclude=set(string.punctuation)

df_2_test['comment_text']=df_2_test['comment_text'].str.replace('[^\w\s]','')
df['comment_text']=df['comment_text'].str.replace('[^\w\s]','')

def clean(doc):
   stop_free=" ".join([i for i in doc.split() if i not in stop])
   punc_free=''.join(i for i in stop_free if i not in exclude )
   num_free=''.join(i for i in punc_free if not i.isdigit())
   return num_free
df_2_test['comment_text']=[clean(df_2_test.iloc[i,1]) for i in range(0,df_2_test.shape[0])]
df['comment_text']=[clean(df.iloc[i,1]) for i in range(0,df.shape[0])]




# In[8]:


# df['comment_text'].head(10)
df_clean=df
df_test_clean=df_2_test


# In[38]:


# from textblob import Word
# from nltk.stem.snowball import SnowballStemmer


# In[39]:


# stemmer=SnowballStemmer("english")
# def stemming(text):
#     text=[stemmer.stem(word) for word in text.split()]
#     return" ".join(text)
# df_clean['comment_text']=df_clean['comment_text'].apply(stemming)


# In[40]:


# from nltk.stem import WordNetLemmatizer
# wlem = WordNetLemmatizer()


# In[41]:


df_clean['comment_text'] = df_clean['comment_text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
df_clean['comment_text'].head()


# In[140]:


# df['comment_text'].head()


# In[ ]:


###############################################################################################################################


# In[ ]:


# wordcloud


# In[113]:


from collections import Counter


# In[142]:


target = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
toxic=df[df.toxic==1]['comment_text'].values
severe_toxic=df[df.severe_toxic==1]['comment_text'].values
obscene=df[df.obscene==1]['comment_text'].values
threat=df[df.threat==1]['comment_text'].values
insult=df[df.insult==1]['comment_text'].values
identity_hate=df[df.identity_hate==1]['comment_text'].values

word_counter = {}
stop=set(stopwords.words('english'))
def clean_text(text):
    stop=set(stopwords.words('english'))
    text = re.sub('[{}]'.format(string.punctuation), ' ', text.lower())
    return ' '.join([word for word in text.split() if word not in (stop)])

for categ in target:
  d = Counter()
  df[df[categ] == 1]['comment_text'].apply(lambda t: d.update(clean_text(t).split()))
  word_counter[categ] = pd.DataFrame.from_dict(d, orient='index')                                        .rename(columns={0: 'count'})                                        .sort_values('count', ascending=False)


word_counter = {}

def clean_text(text):
    text = re.sub('[{}]'.format(string.punctuation), ' ', text.lower())
    return ' '.join([word for word in text.split() if word not in (stop)])

for categ in target:
  d = Counter()
  df[df[categ] == 1]['comment_text'].apply(lambda t: d.update(clean_text(t).split()))
  word_counter[categ] = pd.DataFrame.from_dict(d, orient='index')                                        .rename(columns={0: 'count'})                                        .sort_values('count', ascending=False)
for categ in target:
    print(categ)
    wc = word_counter[categ]
    wordcloud = WordCloud(width = 1000, height = 500,stopwords=STOPWORDS, background_color = 'black').generate_from_frequencies(
                         wc.to_dict()['count'])

    plt.figure(figsize = (15,8))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.title("Words frequented in"+str(categ), fontsize=20)
    plt.show()


# In[ ]:


###############################################################################################################################


# In[ ]:


# Advance preprocessing


# In[30]:


tf calculation
tf1 = (df_clean['comment_text'][0:1]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()
tf1.columns = ['words','tf']
tf1.head(20)


# In[32]:


for i,word in enumerate(tf1['words']):
  tf1.loc[i, 'idf'] = np.log(df_clean.shape[0]/(len(df_clean[df_clean['comment_text'].str.contains(word)])))

tf1.head(5)


# In[34]:


tf1['tfidf'] = tf1['tf'] * tf1['idf']
tf1


# In[44]:


N-gram
TextBlob(df_clean['comment_text'][0]).ngrams(2)


# In[ ]:


# ############################################################################################################################


# In[ ]:


# tf-idf of all and split train test


# In[11]:


# tf-idf vectorizer
word_vectorizer=TfidfVectorizer(ngram_range=(1,4),
                           min_df=3,max_df=0.9,
                           strip_accents='unicode',
                            stop_words = 'english',
                            analyzer = 'word',
                           token_pattern=r'\w{1,}',
                            use_idf=1,
                           smooth_idf=1,
                            sublinear_tf=1,
                            max_features=50000)
vectorizer = word_vectorizer


# In[54]:


vectorizer.fit(df['comment_text'])



# In[13]:


Train_vector_csr=vectorizer.transform(df.comment_text)
Test_vector_csr=vectorizer.transform(df_2_test.comment_text)


# In[15]:


Train_all_text=Train_vector_csr


# In[26]:


x1=df['comment_text']
y1=df.iloc[:,2:8]


# In[27]:


X_train, X_test,y_train,y_test= train_test_split(x1,y1,random_state=42, test_size=0.33)


# In[55]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[29]:


vectorizer.fit(X_train)
X_trainsplit_features=vectorizer.transform(X_train)


# In[30]:


X_testsplit_features=vectorizer.transform(X_test)


# In[189]:


categories = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']


# In[56]:


# NB- model completed:
NB_score=[]
NBauc_score=[]
for i,j in enumerate(categories):
    train_target=y_train[j]
    test_target=y_test[j]
    modelNB=MultinomialNB()
    cv_score=np.mean(cross_val_score(modelNB,X_trainsplit_features,train_target,cv=5,scoring='accuracy'))
    NB_score.append(cv_score)
    modelNB.fit(X_trainsplit_features,train_target)
    y_pred_prob=modelNB.predict_proba(X_testsplit_features)[:,1]
    y_pred = modelNB.predict(X_testsplit_features)
    auc_score=metrics.roc_auc_score(y_test[j],y_pred_prob)
    NBauc_score.append(auc_score)
    print("Accuracy score for class {} is {}".format(j,cv_score))
    print("CV ROC_AUC score {}\n".format(NBauc_score))
    print(classification_report(test_target, y_pred))


# In[57]:


Accuracy_score=pd.DataFrame(index=categories)
Accuracy_score['NB']=NB_score
Accuracy_score


# In[59]:


# LR model completed
LR_score=[]
LRauc_score=[]

    
for i,j in enumerate(categories):
    train_target=y_train[j]
    test_target=y_test[j]
    modelLR=LogisticRegression(C=12)
    cv_score=np.mean(cross_val_score(modelLR,X_trainsplit_features,train_target,cv=5,scoring='accuracy'))
    LR_score.append(cv_score)
    modelLR.fit(X_trainsplit_features,train_target)
    y_pred_prob=modelLR.predict_proba(X_testsplit_features)[:,1]
    y_pred = modelLR.predict(X_testsplit_features)
    auc_score=metrics.roc_auc_score(y_test[j],y_pred_prob)
    LRauc_score.append(auc_score)
    print("Accuracy score for class {} is {}".format(j,cv_score))
    print("CV ROC_AUC score {}\n".format(auc_score))
    print(classification_report(test_target, y_pred))
    


# In[60]:


Accuracy_score['LR']=LR_score
Accuracy_score


# In[ ]:


# NB-LR model completed:
x=X_trainsplit_features
def pr(y_i, y):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)

def get_mdl(y):
    y = y.values
    r = np.log(pr(1,y) / pr(0,y))
    m = LogisticRegression(C=3)
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r


# In[40]:


# preds = np.zeros((len(df_2_test), len(categories)))
NBLR_score=[]
NBLRauc_score=[]
for i, j in enumerate(categories):
    train_target=y_train[j]
    test_target=y_test[j]
    print('fit', j)
    y = y_train[j].values
    r = np.log(pr(1,y) / pr(0,y))
    m = LogisticRegression(C=3,dual=True)
    x_nb = x.multiply(r)
    model=m.fit(x_nb,y_train[j])
#     preds[:,i] = modelLR.predict_proba(Test_vector_csr.multiply(r))[:,1]
    y_pred_prob=m.predict_proba(X_testsplit_features)[:,1]
    y_pred = m.predict(X_testsplit_features)
    cv_score=np.mean(cross_val_score(m,x_nb,train_target,cv=5,scoring='accuracy'))
    NBLR_score.append(cv_score)
    auc_score=metrics.roc_auc_score(y_test[j], y_pred_prob)
    NBLRauc_score.append(auc_score)
    print("Accuracy score for class {} is {}".format(j,cv_score))
    print("CV ROC_AUC score {}\n".format(auc_score))
    print(classification_report(test_target, y_pred))
#     submission file.

#     submid = pd.DataFrame({'id': subm["id"]})
#     submission = pd.concat([submid, pd.DataFrame(preds, columns = categories)], axis=1)
#     submission.to_csv('submission.csv', index=False)


# In[53]:


Accuracy_score['NB-LR']=NBLR_score
Accuracy_score


# In[22]:


a=Test_vector_csr
b=Train_vector_csr


# In[23]:


# NB-LR model submission file:
x=Train_all_text
def pr(y_i, y):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)
x = b
test_x = a

def get_mdl(y):
    y = y.values
    r = np.log(pr(1,y) / pr(0,y))
    m = LogisticRegression(C=3)
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r



# In[24]:


categories = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']


# In[25]:


# Final submissiom
preds = np.zeros((len(df_2_test), len(categories)))
NBLR_score=[]
NBLRauc_score=[]
for i, j in enumerate(categories):
    train_target=y_train[j]
    test_target=y_test[j]
    print('fit', j)
    m,r = get_mdl(df[j])
    preds[:,i] = m.predict_proba(test_x.multiply(r))[:,1]
# #     y_pred_prob=m.predict_proba(X_testsplit_features)[:,1]
# #     y_pred = m.predict(X_testsplit_features)
# #     cv_score=np.mean(cross_val_score(m,X_trainsplit_features,train_target,cv=5,scoring='accuracy'))
#     NBLR_score.append(cv_score)
#     auc_score=metrics.roc_auc_score(y_test[j], y_pred_prob)
#     NBLRauc_score.append(auc_score)
#     print("Accuracy score for class {} is {}".format(j,cv_score))
#     print("CV ROC_AUC score {}\n".format(auc_score))
#     print(classification_report(test_target, y_pred))
#     And finally, create the submission file.


    submid = pd.DataFrame({'id': subm["id"]})
    submission = pd.concat([submid, pd.DataFrame(preds, columns = categories)], axis=1)
    submission.to_csv('submission.csv', index=False)
#     submid = pd.DataFrame({'id': subm["id"]})
#     submission = pd.concat([submid, pd.DataFrame(preds, columns = categories)], axis=1)
#     submission.to_csv('submission.csv', index=False)

