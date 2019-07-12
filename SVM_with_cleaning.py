import pandas as pd
import numpy as np
import re
import nltk
import time
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from nltk.corpus import stopwords
from sklearn import svm 
df = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
#print df.head()
df.dropna(inplace=True)
df["Postive rated"]=np.where(df['sentiment']>0,1,0)
def cleaning_words(raw_words):
    exam=BeautifulSoup(raw_words,"html.parser") #removing html tags
    letters=re.sub("[^a-zA-Z]"," ",exam.get_text()) #removing numbers and others except small and capital alphabets
    low=letters.lower() #Converting everything to lower case
    words=low.split() #spiliting sentences into words
    useful= [w for w in words if not w in stopwords.words("english")] #removing stopping words
    use_sent= " ".join(useful)
    return use_sent
num=df["review"].size
#print num
perfect_words=[]

for i in range(0,num):
    #if( (i+1)%1000 == 0 ):
    print ("Review %d of %d\n"%(i+1, num))
    perfect_words.append(cleaning_words(df["review"][i]))

#print df.head(67)
#print df["Postive rated"].mean()
X_train, X_test, y_train, y_test = train_test_split(perfect_words,df['Postive rated'],random_state=0)

vect=CountVectorizer(min_df=5,ngram_range=(1,2)).fit(X_train) #words converting into vectors , minimum document frequency min_df=5
print (len(vect.get_feature_names()))

X_train_vetorised=vect.transform(X_train)

clf = svm.SVC(kernel='linear', C = 1.0)

#clf.fit(x,y)
clf.fit(X_train_vetorised,y_train)

predictions=clf.predict(vect.transform(X_test))

print ("AUC:",roc_auc_score(y_test,predictions))

