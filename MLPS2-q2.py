#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score


# Store the digits dataset
digits = datasets.load_digits()
print(digits.data.shape) # (1797, 64)

# Create Decision Tree Classifer object with default parameters
tree = DecisionTreeClassifier()

# Train Decision Tree Classifer
model = tree.fit(digits.data, digits.target)

# Report classification accuracy score
accuracy_score = model.score(digits.data, digits.target)
print(accuracy_score)


# # Part B
# 
# Write a Python program that uses the sklearn cross_val_score function to evaluate the classification accuracy of the DecisionTreeClassifier model on the digits data set as in the preceding part. Do this once for each of the following numbers of cross-validation folds: 2^n, where n = 1, 2, 3, 4, 5, 6, 7. Your program should report the mean classification accuracy for each number of folds, and plot the mean accuracy as a function of the number of folds. 
# 
# Use matplotlib.pyplot.plot for plotting. Include comments that explains the steps in the computation. Submit the text of your source code in your main writeup, and attach the .py source file separately.

# In[104]:


# Create Decision Tree Classifer object with default parameters
tree = DecisionTreeClassifier()

# Train Decision Tree Classifer
model = tree.fit(digits.data, digits.target)

# Initialize list of cross-validation folds, which equals 2^n, where n = [1,7]
folds = [(2 ** i) for i in range(1, 8)]

# Initialize empty list for mean accuracy
accuracy = []

# Calculate classification accuracy with varying number of cross-validation folds
for num in folds:
    cv_score = cross_val_score(model, digits.data, digits.target, cv=num)
    
    # Calculate average, then append 
    mean_accuracy = np.mean(cv_score)
    accuracy.append(mean_accuracy)


# In[105]:


plt.scatter(folds, accuracy)
plt.title("Number of cross-validation folds vs. Accuracy score")
plt.show()

