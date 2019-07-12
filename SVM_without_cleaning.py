import pandas as pd
import numpy as np
from sklearn import svm 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
df = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
df.dropna(inplace=True)
df["Postive rated"]=np.where(df['sentiment']>0,1,0)
df.head()
X_train, X_test, y_train, y_test = train_test_split(df['review'],df['Postive rated'],random_state=0)
vect=CountVectorizer(min_df=5,ngram_range=(1,2)).fit(X_train)
#print(vect)
#print(X_train)
X_train_vetorised=vect.transform(X_train)
#print (X_train_vetorised) 
clf = svm.SVC(kernel='linear', C = 1.0)

#clf.fit(x,y)
clf.fit(X_train_vetorised,y_train)

predictions=clf.predict(vect.transform(X_test))

print ("AUC:",roc_auc_score(y_test,predictions))

