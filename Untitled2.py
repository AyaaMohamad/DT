#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import seaborn as sns


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


from sklearn.model_selection import train_test_split


# In[5]:


from sklearn.tree import DecisionTreeClassifier, plot_tree


# In[6]:


from sklearn.neighbors import KNeighborsClassifier


# In[7]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[8]:


# Load the dataset
data = pd.read_csv('diabetes.csv')


# In[9]:


# Display the first few rows of the dataset
print(data.head())


# In[10]:


# Display basic statistics
print(data.describe())


# In[11]:


# Check for missing values
print(data.isnull().sum())


# In[12]:


# Visualize the distribution of each feature
data.hist(figsize=(12, 10))
plt.tight_layout()
plt.show()


# In[13]:


# Pairplot of the features
sns.pairplot(data, hue='Outcome')
plt.show()


# In[14]:


# Split the data into features and target variable
X = data.drop('Outcome', axis=1)
y = data['Outcome']


# In[15]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[16]:


# Initialize the Decision Tree classifier
dt_classifier = DecisionTreeClassifier(random_state=42)


# In[17]:


# Train the model
dt_classifier.fit(X_train, y_train)


# In[18]:


# Make predictions
y_pred_dt = dt_classifier.predict(X_test)


# In[19]:


# Evaluate the model
print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred_dt)}")
print(f"Decision Tree Classification Report:\n {classification_report(y_test, y_pred_dt)}")


# In[20]:


# Plot confusion matrix for Decision Tree
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_dt, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Diabetic', 'Diabetic'], yticklabels=['Non-Diabetic', 'Diabetic'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Decision Tree Confusion Matrix')
plt.show()


# In[21]:


# Plot decision tree
plt.figure(figsize=(20, 10))
plot_tree(dt_classifier, feature_names=X.columns, class_names=['Non-Diabetic', 'Diabetic'], filled=True)
plt.show()


# In[22]:


# Initialize the KNN classifier with k=5
knn_classifier = KNeighborsClassifier(n_neighbors=5)


# In[23]:


# Train the model
knn_classifier.fit(X_train, y_train)


# In[24]:


# Make predictions
y_pred_knn = knn_classifier.predict(X_test)


# In[25]:


# Evaluate the model
print(f"KNN Accuracy: {accuracy_score(y_test, y_pred_knn)}")
print(f"KNN Classification Report:\n {classification_report(y_test, y_pred_knn)}")


# In[26]:


# Plot confusion matrix for KNN
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_knn, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Diabetic', 'Diabetic'], yticklabels=['Non-Diabetic', 'Diabetic'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('KNN Confusion Matrix')
plt.show()


# In[ ]:




