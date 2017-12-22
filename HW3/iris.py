import pandas as pd
import numpy as np
import sklearn
from sklearn import tree
from sklearn.cross_validation import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv('iris.csv', header=None)

df = sklearn.utils.shuffle(df)
#df = df.reset_index(drop=True)
#print df
df_data = df.iloc[:,0:4]
df_target = df.iloc[:,4]
###turn into numpy type
df_data = df_data.values
df_target = df_target.values
#print df_data
x_train, x_test, y_train, y_test = train_test_split(df_data, df_target, test_size=0.3)

#print y_test

clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)
#print clf.feature_importances_ 
print 'DecisionTreeClassifier:'
print 'mean square error',clf.score(x_test,y_test)
print "recall",recall_score(y_test,clf.predict(x_test),average='macro')
print "precision",precision_score(y_test,clf.predict(x_test),average='macro')

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(x_train, y_train) 
print '\nKNeighborsClassifier:'
print 'mean square error',neigh.score(x_test,y_test)
print "recall",recall_score(y_test,neigh.predict(x_test),average='macro')
print "precision",precision_score(y_test,neigh.predict(x_test),average='macro')

gnb = GaussianNB()
y_pred = gnb.fit(x_train, y_train)
print '\nGaussianNB:'
print 'mean square error',gnb.score(x_test,y_test)
print "recall",recall_score(y_test,gnb.predict(x_test),average='macro')
print "precision",precision_score(y_test,gnb.predict(x_test),average='macro')
