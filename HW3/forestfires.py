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
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

df = pd.read_csv('forestfires.csv')

plt.figure(figsize=(20,10))
plt.title('Area',fontsize=30)
plt.plot(df['area'])
plt.show()

df['month'].replace( ('jan','feb','mar','apr','may', 'jun','jul','aug','sep', 'oct', 'nov','dec'),(1,2,3,4,5,6,7,8,9,10,11,12),inplace=True)
df['day'].replace( ('fri', 'tue', 'sat', 'sun', 'mon', 'wed', 'thu'),(5,2,6,7,1,3,4),inplace=True)
df_data = df.drop(['area'],axis=1)
df_target = df['area']

x_train, x_test, y_train, y_test = train_test_split(df_data, df_target, test_size=0.3)

regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x_train,y_train)

index = np.arange(0, len(x_test),1)
mse = mean_squared_error(y_test,regressor.predict(x_test)) 
print 'mean square error: ',mse

plt.figure(figsize=(20,10))
plt.title('DecisionTreeRegressor',fontsize=30)
plt.plot(index,regressor.predict(x_test),label="Prediction (MSE:%0.2f)" %mse)
plt.plot(index,y_test,c='r',label="ground truth")
plt.legend(fontsize = 20)
plt.show()
print 'mean accuracy:',regressor.score(x_test,y_test)

#####################
###KNeighborsRegressor
neigh = KNeighborsRegressor(n_neighbors=3)
neigh.fit(x_train,y_train)
print 'mean square error :',neigh.score(x_test,y_test)

mse = mean_squared_error(y_test,neigh.predict(x_test)) 
print 'mean square error: ',mse

plt.figure(figsize=(15,10))
plt.title('KNeighborsRegressor k=3',fontsize=30)
plt.plot(index,neigh.predict(x_test),label="Prediction (MSE:%0.2f)" %mse)
plt.plot(index,y_test,c='r',label="ground truth")
plt.legend(fontsize = 20)
plt.show()

#####################
###Naive Bayes
def area_to_class(y):
    if y==0:
        return 0
    elif y>0 and y<=1:
        return 1
    elif y>1 and y<=10:
        return 2
    elif y>10 and y<=100:
        return 3
    elif y>100 and y<=1000:
        return 4
    elif y>1000:
        return 5
    
y_train_class = y_train.apply(area_to_class)
y_test_class = y_test.apply(area_to_class)
y_train_class.hist(figsize = (9,5), label='train')
y_test_class.hist(figsize = (9,5), label = 'test')
plt.legend(fontsize = 15)

clf = GaussianNB()
clf.fit(x_train, y_train_class)
print clf.score(x_test, y_test_class)
print clf.predict(x_test)
plt.hist(clf.predict(x_test),bins=[0,1,2,3,4,5],rwidth=0.5,align='left')
plt.show()

###Define categorical features & continuous features
x_cate_train = x_train[['X','Y','month','day']]
x_conti_train = x_train[['FFMC','DMC','DC','ISI','temp','RH','wind','rain']]
x_cate_test = x_test[['X','Y','month','day']]
x_conti_test = x_test[['FFMC','DMC','DC','ISI','temp','RH','wind','rain']]

clf_Gaussian = GaussianNB()
clf_Gaussian.fit(x_conti_train, y_train_class)
print 'clf_Gaussian mean square error:',clf_Gaussian.score(x_conti_test, y_test_class)

clf_Laplace = MultinomialNB()
clf_Laplace.fit(x_cate_train, y_train_class)
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
print 'clf_Laplace mean square error:',clf_Laplace.score(x_cate_test, y_test_class)

y_Laplace = clf_Laplace.predict_proba(x_cate_test).argmax(1)
print y_Laplace
plt.hist(y_Laplace,bins=[0,1,2,3,4,5],rwidth=0.5,align='left')
plt.title('y_Laplace')
plt.show()

y_Gaussian = clf_Gaussian.predict_proba(x_conti_test).argmax(1)
print y_Gaussian
plt.hist(y_Gaussian,bins=[0,1,2,3,4,5],rwidth=0.5,align='left')
plt.title('y_Gaussian')
plt.show()

#print clf_Laplace.predict_proba(x_cate_test)
y_pred = (clf_Gaussian.predict_proba(x_conti_test)*clf_Laplace.predict_proba(x_cate_test)/clf_Gaussian.class_prior_).argmax(1)

print 'MultinomialNB model score:',accuracy_score(y_test_class, y_Laplace)
print 'GaussianNB model score:',accuracy_score(y_test_class, y_Gaussian)
print 'Mixed model score:',accuracy_score(y_test_class, y_pred)