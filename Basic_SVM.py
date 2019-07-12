import numpy as np
from sklearn import svm
x =np.array([[1,2],
             [5,8],
             [1.5,1.8],
             [8,8],
             [1,0.6],
             [9,11]])
y = [0,1,0,1,0,1]

clf = svm.SVC(kernel='linear', C = 1.0)
clf.fit(x,y)

print (clf.predict([[0.1,0.76]]))
print (clf.predict([[5.1,3.0]]))
