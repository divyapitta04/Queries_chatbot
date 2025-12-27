import num2words
import nltk
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
from nltk.stem.porter import *
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english')) 
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
stemmer=PorterStemmer()
from gensim.models import Word2Vec
import pandas as pd
from numpy import dot
from numpy.linalg import norm

df=pd.read_csv('CancerQA.csv',encoding='unicode_escape')
df_combined = df

def preprocess_text(text):
    lemmatizer=WordNetLemmatizer()
    text=text.lower()
    token1=tokenizer.tokenize(text)
    token=[]
    for x in token1:
        if x not in stop_words:
            token.append(x)
            #stemmed = [stemmer.stem(tokens) ]
    lemmatiz=[lemmatizer.lemmatize(tokens) for tokens in token]
    return lemmatiz


# ## preprocessing
topics=[[] for i in range(len(df))]
para=[[] for i in range(len(df))]
topics1=[[] for i in range(len(df))]
para1=[[] for i in range(len(df))]
for i in range(len(df_combined)):
    text=df_combined.iloc[i][0]
    text=str(text)
    topics[i]=preprocess_text(text)
    text=df_combined.iloc[i][1]
    text=str(text)
    para[i]=preprocess_text(text)

def Answer_Pre_Processing(file):
    token_files=[]
    tokenizer = RegexpTokenizer(r'\w+')
    token_files=(tokenizer.tokenize(str(file)))
    for i in range(len(token_files)):
        if(token_files[i]=='u' or token_files[i]=='U'):
            token_files[i]='you'
        elif(token_files[i]=='d' ):
            token_files[i]='the'
        elif(token_files[i]=='n' or token_files[i]=='nd'):
            token_files[i]='and'
        elif(token_files[i]=='hv' ):
            token_files[i]='have'
        elif(token_files[i]=='bcoz' or token_files[i]=='becoz' or token_files=='bcz'):
            token_files[i]='because'
        elif(token_files[i]=='ur' or token_files[i]=='Ur'):
            token_files[i]='your'
        elif(token_files[i]=='thru'):
            token_files[i]='through'
    str1=""
    for word in token_files:
        str1=str1+word+" "
    return str1


# ## applying model
model_para=Word2Vec(para, min_count=1)
model_para.init_sims(replace=True)
model_topic=Word2Vec(topics,min_count=1)
model_topic.init_sims(replace=True)

def sentence_vector(model, sentence):
    # Average word vectors for a sentence
    vectors = [model.wv[word] for word in sentence if word in model.wv]
    if not vectors:
        return None
    return sum(vectors) / len(vectors)

def cosine_sim(a, b):
    return dot(a, b) / (norm(a) * norm(b))

def get_answers(df_combined, query1):
    query = Answer_Pre_Processing(query1)
    q = preprocess_text(query)
    q_vec = sentence_vector(model_para, q)
    if q_vec is None:
        return "Sorry, I couldn't understand your query.", 0

    max_sim = -1
    result = ""
    for i in range(len(df_combined)):
        para_vec = sentence_vector(model_para, para[i])
        if para_vec is None:
            continue
        sim = cosine_sim(q_vec, para_vec)
        if sim > max_sim:
            max_sim = sim
            print("The answer similarity is:", max_sim)
            print("The answer is:", df_combined.iloc[i]['Answer'])
            result = df_combined.iloc[i]['Answer']
    return result, max_sim


# ## initiating chat process
print("Hello user")
print("How may I help you")
print("For exiting from the chatbot,press 0")
x=2
while x!=0:
    print('\n \n')
    print('Enter query')
    query=input()
    q=preprocess_text(query)
    
    q1=""
    for d in q:
        q1=q1+d+" "
    res,r=get_answers(df,q1)
    print(r)
    if r<2:
        print(res)
    else:
        print("Sorry I don't have the answer. Can you please rephrase the query")
    print("If you have any more queries then press 1 else press 0")
    x=int(input())
    if x==0:
        print("Thank you for using the chatbot.I hope you had a great time")

