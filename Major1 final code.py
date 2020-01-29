#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib as plt
import numpy as np
import seaborn as sn


# In[2]:


dataset=pd.read_csv('C:\\Users\\Ravi Singh\\Desktop\\churndataset\\churn_data.csv')


# In[3]:


dataset.head()


# In[4]:


dataset.columns


# In[5]:


dataset.describe()


# In[6]:


#cleaning data
# removing NaN
dataset.isna().any()


# In[7]:


dataset.isna().sum()


# In[8]:


dataset=dataset[pd.notnull(dataset['age'])]
dataset=dataset.drop(columns=['credit_score','rewards_earned'])


# In[9]:


#histograms
dataset2=dataset.drop(columns=['user','churn'])


# In[10]:


from matplotlib import pyplot as plt


# In[11]:


#ploting figure
fig = plt.figure(figsize=(15,12))
plt.suptitle('Histograms of Numerical Columns', fontsize=20)
for i in range(1,dataset2.shape[1]+1):
    plt.subplot(6,5,i)
    f=plt.gca()
    f.axes.get_yaxis().set_visible(False)
    f.set_title(dataset2.columns.values[i-1])
    vals=np.size(dataset2.iloc[:,i-1].unique())
    plt.hist(dataset2.iloc[:,i-1],bins=vals,color='#3F5D7D')
plt.tight_layout(rect=[0,0.03,1,0.95])


# In[12]:


#piechart plots
dataset2=dataset[['housing','is_referred','app_downloaded','web_user', 'app_web_user', 'ios_user',
       'android_user', 'registered_phones', 'payment_type', 'waiting_4_loan',
       'cancelled_loan', 'received_loan', 'rejected_loan', 'zodiac_sign',
       'left_for_two_month_plus', 'left_for_one_month', 'is_referred']]


# In[13]:


#ploting piechart
fig = plt.figure(figsize=(15,12))
plt.suptitle('Pie Chart Distributions', fontsize=20)
for i in range(1,dataset2.shape[1]+1):
    plt.subplot(6,3,i)
    f=plt.gca()
    f.axes.get_yaxis().set_visible(False)
    f.set_title(dataset2.columns.values[i-1])
    values=dataset2.iloc[:,i-1].value_counts(normalize=True).values
    index=dataset2.iloc[:,i-1].value_counts(normalize=True).index
    
    plt.pie(values,labels=index,autopct='%1.1f%%')
plt.tight_layout(rect=[0,0.03,1,0.95])


# In[14]:


## Exploring Uneven Features
dataset[dataset2.waiting_4_loan==1].churn.value_counts()


# In[15]:


dataset[dataset2.cancelled_loan == 1].churn.value_counts()


# In[16]:


dataset[dataset2.received_loan == 1].churn.value_counts()


# In[17]:


dataset[dataset2.rejected_loan == 1].churn.value_counts()


# In[18]:


dataset[dataset2.left_for_one_month == 1].churn.value_counts()


# In[19]:


#correlation plot
dataset.drop(columns = ['churn','user','housing', 'payment_type',
                         'registered_phones', 'zodiac_sign']).corrwith(dataset.churn).plot.bar(
    figsize = (20, 10), title = 'Correlation with the Response Variable', fontsize = 15,
        rot = 45, grid = True)


# In[20]:


# Correlation Matrix
sn.set(style="white")

# Compute the correlation matrix
corr = dataset.drop(columns=['user','churn']).corr()


# In[21]:


# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(18, 15))

# Generate a custom diverging colormap
cmap = sn.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sn.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}) 


# In[22]:


dataset=dataset.drop(columns = ['app_web_user'])


# In[25]:


dataset.to_csv('new_churn_data.csv',index=False)


# In[24]:


dataset=pd.read_csv('new_churn_data.csv')


# In[26]:


user_identifier = dataset['user']
dataset= dataset.drop(columns = ['user'])


# In[27]:


# One-Hot Encoding
dataset.housing.value_counts()


# In[28]:


dataset = pd.get_dummies(dataset)
dataset.columns


# In[29]:


dataset = dataset.drop(columns = ['housing_na', 'zodiac_sign_na', 'payment_type_na'])
dataset.info()


# In[30]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset.drop(columns = 'churn'), dataset['churn'],
                                                    test_size = 0.2,
                                                    random_state = 0)


# In[31]:


y_train.value_counts()


# In[33]:


import random


# In[34]:


# Balancing the Training Set
pos_index = y_train[y_train.values == 1].index
neg_index = y_train[y_train.values == 0].index

if len(pos_index) > len(neg_index):
    higher = pos_index
    lower = neg_index
else:
    higher = neg_index
    lower = pos_index

random.seed(0)
higher = np.random.choice(higher, size=len(lower))
lower = np.asarray(lower)
new_indexes = np.concatenate((lower, higher))

X_train = X_train.loc[new_indexes,]
y_train = y_train[new_indexes]


# In[37]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train2 = pd.DataFrame(sc_X.fit_transform(X_train))
X_test2 = pd.DataFrame(sc_X.transform(X_test))
X_train2.columns = X_train.columns.values
X_test.columns = X_test.columns.values
X_train2.index = X_train.index.values
X_test.index = X_test.index
X_train = X_train2
X_test = X_test2


# In[42]:


### Comparing Models

## Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state =0)
classifier.fit(X_train, y_train)
# Predicting Test Set
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

results = pd.DataFrame([['Linear Regression (Lasso)', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])


# In[43]:


results


# In[44]:


## SVM (rbf)
from sklearn.svm import SVC
classifier = SVC(random_state = 0, kernel = 'rbf')
classifier.fit(X_train, y_train)

# Predicting Test Set
y_pred = classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results = pd.DataFrame([['SVM (RBF)', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

results = results.append(model_results, ignore_index = True)


# In[45]:


## Randomforest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(random_state = 0, n_estimators = 100,
                                    criterion = 'entropy')
classifier.fit(X_train, y_train)

# Predicting Test Set
y_pred = classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results = pd.DataFrame([['Random Forest (n=100)', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

results = results.append(model_results, ignore_index = True)


# In[46]:


results


# In[47]:


## Confusion Matrix
df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, fmt='g')
print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred))


# In[48]:


## K-fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X= X_train, y = y_train,
                             cv = 10)


# In[49]:


accuracies


# In[50]:


accuracies.mean()


# In[74]:


#### Feature Selection ####


## Feature Selection
# Recursive Feature Elimination
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()


# In[75]:


X_train.shape


# In[76]:


# Select Best X Features
rfe = RFE(classifier, 20)
rfe = rfe.fit(X_train, y_train)


# In[77]:


# summarize the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)
X_train.columns[rfe.support_]


# In[78]:


corr = X_train[X_train.columns[rfe.support_]].corr()


# In[79]:


# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(18, 15))

# Generate a custom diverging colormap
cmap = sn.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sn.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[81]:


## Randomforest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(random_state = 0, n_estimators = 100,
                                    criterion = 'entropy')
classifier.fit(X_train[X_train.columns[rfe.support_]], y_train)

# Predicting Test Set
y_pred = classifier.predict(X_test[X_test.columns[rfe.support_]])
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results = pd.DataFrame([['Random Forest after new Feature selection', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

results = results.append(model_results, ignore_index = True)


# In[82]:


results


# In[83]:


## K-fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X= X_train, y = y_train,
                             cv = 10)
print("Random Forest Classifier Accuracy: %0.2f (+/- %0.2f)"  % (accuracies.mean(), accuracies.std() * 2))


# In[84]:


# Formatting Final Results
final_results = pd.concat([y_test, user_identifier], axis = 1).dropna()
final_results['predicted_churn'] = y_pred
final_results = final_results[['user', 'churn', 'predicted_churn']].reset_index(drop=True)


# In[89]:


final_results


# In[ ]:




