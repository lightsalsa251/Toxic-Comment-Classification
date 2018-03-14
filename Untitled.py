
# coding: utf-8

# In[1]:


import nltk


# In[2]:


nltk.download()


# In[3]:


get_ipython().system('wget https://github.com/shbm/kaggle_toxic_comments/blob/master/dataset/train.csv.zip?raw=true')


# In[4]:


get_ipython().system('wget https://github.com/shbm/kaggle_toxic_comments/blob/master/dataset/test.csv.zip?raw=true')


# In[5]:


ls


# In[6]:


get_ipython().system('unzip test.csv.zip?raw=true')
get_ipython().system('unzip train.csv.zip?raw=true')


# In[8]:


import pandas as pd
test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')


# In[9]:


from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re


# In[10]:


def clean(text):
    text = str(text)
    text = text.split(',')
    text = " ".join([w.lower() for w in text])

    text = re.sub(r'[^A-Za-z\s]',r' ',text)
    text = re.sub(r'\n',r' ',text)    
    
    lemmatizer = WordNetLemmatizer()
    for i in text:
        i = lemmatizer.lemmatize(i)

    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    text = " ".join([w for w in words if not w in stop_words])
    
    return text
    
    


# In[11]:


train = train['comment_text']
test = test['comment_text']
test = list(test)
train = list(train)
train = train+test
len(train)


# In[13]:


train = list(map(clean,train))
print(train[0])


# In[ ]:


COMMENT = 'comment_text'
train[COMMENT].fillna("unknown", inplace=True)


# In[ ]:


train_features = train[:159571]
test_features = train[159571:]


# In[ ]:


vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1 )
trn_term_doc = vec.fit_transform(train_features[COMMENT])
test_term_doc = vec.transform(test_features[COMMENT])


# In[ ]:


label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


# In[ ]:


def pr(y_i, y):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)


# In[ ]:


x = trn_term_doc
test_x = test_term_doc


# In[ ]:


def get_mdl(y):
    y = y.values
    r = np.log(pr(1,y) / pr(0,y))
    m = LogisticRegression(C=4, dual=True)
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r


# In[ ]:


preds = np.zeros((len(test), len(label_cols)))

for i, j in enumerate(label_cols):
    print('fit', j)
    m,r = get_mdl(train[j])
    preds[:,i] = m.predict_proba(test_x.multiply(r))[:,1]

