#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("train.csv")
df.head()


# In[3]:


df.tail()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:


df.describe() 


# In[8]:


df['ApplicantIncome']


# In[9]:


df[['ApplicantIncome', 'LoanAmount']]


# In[10]:


df.columns


# In[11]:


df.isnull().sum()


# In[12]:


df.info()


# In[13]:


# handle numerical missing data
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean())
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mean())


# In[14]:


df.isnull().sum()


# In[15]:


# handle categorical missing data
df['Gender'].mode()[0]


# In[16]:


df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])


# In[17]:


df.isnull().sum()


# In[18]:


# categorical data
import seaborn as sns


# In[19]:


sns.countplot(x='Gender', data=df)


# In[20]:


df['Dependents'] = df['Dependents'].replace('3+', 3)
sns.countplot(df.Dependents)


# In[21]:


df.columns


# In[22]:


# numerical data
sns.displot(df, x="ApplicantIncome", kde=True)


# In[23]:


sns.histplot(data=df, x="LoanAmount")


# In[24]:


sns.displot(df.Credit_History, kde=True)


# In[25]:


df.head()


# In[26]:


# created new column


# In[27]:


df['Total_income'] = df['ApplicantIncome']+df['CoapplicantIncome']


# In[28]:


df.head()


# In[29]:


# data transformation


# In[30]:


df['ApplicantIncomeLog'] = np.log(df['ApplicantIncome'])
df.head()


# In[31]:


sns.displot(df['ApplicantIncomeLog'], kde=True)
plt.show()


# In[32]:


df['CoapplicantIncomeLog'] = np.log(df['CoapplicantIncome'] + 1e-10)
sns.histplot(df["ApplicantIncomeLog"], kde=True)
plt.show()


# In[33]:


df['LoanAmountLog'] = np.log(df['LoanAmount'])
sns.displot(df["LoanAmountLog"], kde=True)


# In[34]:


df['Loan_Amount_Term_Log'] = np.log(df['Loan_Amount_Term'])
sns.histplot(df["Loan_Amount_Term_Log"] , kde=True)


# In[35]:


df['Total_Income_Log'] = np.log(df['Total_income'])
sns.histplot(df["Total_Income_Log"])


# In[36]:


df.head()


# In[37]:


cols = ['ApplicantIncome', 'CoapplicantIncome', "LoanAmount", "Loan_Amount_Term", "Total_income", 'Loan_ID', 'CoapplicantIncomeLog']
df = df.drop(columns=cols, axis=1)


# In[38]:


df.head()


# In[39]:


df.Loan_Status.value_counts()


# In[40]:


df.info()


# In[41]:


df.Education.value_counts()


# In[42]:


df.info()


# In[43]:


df.head()


# In[44]:


d1 = pd.get_dummies(df['Gender'], drop_first= True)
d2 = pd.get_dummies(df['Married'], drop_first= True)
d3 = pd.get_dummies(df['Dependents'], drop_first= True)
d4 = pd.get_dummies(df['Education'], drop_first= True)
d5 = pd.get_dummies(df['Self_Employed'], drop_first= True)
d6 = pd.get_dummies(df['Property_Area'], drop_first= True)



df1 = pd.concat([df, d1, d2, d3, d4, d5, d6], axis = 1)
df=df1

cols = ['Gender', 'Married', "Dependents", "Education", "Self_Employed", 'Property_Area']
df = df.drop(columns=cols, axis=1)


# In[45]:


# cols = ['Gender',"Married","Education",'Self_Employed',"Property_Area","Loan_Status","Dependents"]
# for col in cols:
#     df[col] = pd.get_dummies(df[col], drop_first= True)


# In[46]:


df.head()


# In[47]:


df.info()


# In[48]:


df.describe()


# In[49]:


# test datasets


# In[50]:


test = pd.read_csv("test.csv")

# filling numerical missing data
test['LoanAmount'] = test['LoanAmount'].fillna(test['LoanAmount'].mean())
test['Loan_Amount_Term'] = test['Loan_Amount_Term'].fillna(test['Loan_Amount_Term'].mean())
test['Credit_History'] = test['Credit_History'].fillna(test['Credit_History'].mean())

# filling categorical missing data
test['Gender'] = test['Gender'].fillna(test['Gender'].mode()[0])
test['Married'] = test['Married'].fillna(test['Married'].mode()[0])
test['Dependents'] = test['Dependents'].fillna(test['Dependents'].mode()[0])
test['Self_Employed'] = test['Self_Employed'].fillna(test['Self_Employed'].mode()[0])

test['Total_income'] = test['ApplicantIncome'] + test['CoapplicantIncome']

# apply log transformation to the attribute
test['ApplicantIncomeLog'] = np.log(test['ApplicantIncome'])

test['CoapplicantIncomeLog'] = np.log(test['CoapplicantIncome'] + 1e-10)

test['LoanAmountLog'] = np.log(test['LoanAmount'] + 1e-10)

test['Loan_Amount_Term_Log'] = np.log(test['Loan_Amount_Term'] +1e-10)

test['Total_Income_Log'] = np.log(test['Total_income'])

cols = ['ApplicantIncome', 'CoapplicantIncome', "LoanAmount", "Loan_Amount_Term", "Total_income", 'Loan_ID', 'CoapplicantIncomeLog']
test = test.drop(columns=cols, axis=1)

t1 = pd.get_dummies(test['Gender'], drop_first=True)
t2 = pd.get_dummies(test['Married'], drop_first=True)
t3 = pd.get_dummies(test['Dependents'], drop_first=True)
t4 = pd.get_dummies(test['Education'], drop_first=True)
t5 = pd.get_dummies(test['Self_Employed'], drop_first=True)
t6 = pd.get_dummies(test['Property_Area'], drop_first=True)

df1 = pd.concat([test, t1, t2, t3, t4, t5, t6], axis=1)
test = df1

cols = ['Gender', 'Married', "Dependents", "Education", "Self_Employed", 'Property_Area']
test = test.drop(columns=cols, axis=1)



# In[51]:


test.head()


# In[52]:


df.head()


# In[53]:


# specify input and output attributes
x = df.drop(columns=['Loan_Status'], axis=1)
y = df['Loan_Status']


# In[54]:


y


# In[55]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)


# In[56]:


x_train.head()


# In[57]:


y_test.head()


# In[58]:


# randomforest classifier
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

model.fit(x_train, y_train)


# In[59]:


print("Accuracy is", model.score(x_test, y_test)*100)


# In[60]:


# decision tree classifier
from sklearn.tree import DecisionTreeClassifier
model2 = DecisionTreeClassifier()
model2.fit(x_train, y_train)


print("Accuracy is", model2.score(x_test, y_test)*100)


# In[61]:


# logistic regression
from sklearn.linear_model import LogisticRegression
model3 = LogisticRegression(max_iter=1000)
model3.fit(x_train, y_train)


print("Accuracy is", model3.score(x_test, y_test)*100)


# In[62]:


# confusion matrics


# In[63]:


# random forest classifier
from sklearn.metrics import confusion_matrix
y_pred = model.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
cm


# In[64]:


# model save


# In[65]:


import pickle
file=open("model.pkl", 'wb')
pickle.dump(model, file)

