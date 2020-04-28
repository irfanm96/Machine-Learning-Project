import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
# from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

#Read Training dataset
data = pd.read_csv('./input/data.csv', delimiter=',')
test_data = pd.read_csv('./input/testdata.csv', delimiter=',')

#Get the labels
# label = data['A16'].tolist()
#Remove the labels from the data frame
# data.drop(['A16'], axis=1, inplace=True)

#Replacing non-standard data with NaN
data = data.replace( "?" , np.NaN)
test_data = test_data.replace( "?" , np.NaN)

#Replacing NaN values in numeric variables with mean values for that variable
data.fillna(data.mean(), inplace=True)
test_data.fillna(data.mean(), inplace=True)

#Creating a loop to run on character variables to replace NaN values with most frequent values in that variable
for i in data:
    #If the variable is object type
    if data[i].dtypes == 'object':
        # replace with the most frequent value
        data = data.fillna(data[i].value_counts().index[0])

for i in test_data:
    #If the variable is object type
    if test_data[i].dtypes == 'object':
        # replace with the most frequent value
        test_data = test_data.fillna(test_data[i].value_counts().index[0])

# #Check if any variables have any missing values
# print(data.isna().sum())
# print(test_data.isna().sum())

#Converting non-numeric data into numeric data using LabelEncoder
encode = preprocessing.LabelEncoder()

#Extracting data types for each column and performing numeric transformation
for i in data:
    if data[i].dtypes =='object' or data[i].dtypes =='bool' :
        data[i]=encode.fit_transform(data[i])

for i in test_data:
    if test_data[i].dtypes =='object' or test_data[i].dtypes =='bool' :
        test_data[i]=encode.fit_transform(test_data[i])

# #Lets check if transformation has worked correctly
# print(data.tail(20))
# print(test_data.tail(20))

#Creating correlation matrix for our data to check correlation of variables and possibility of multicollinearity
corr = data.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

#We can observe feature A4,A3 highly correlated, this might lead to multicollinearity
#we can also observe that feature A1 have very less correlation with variable A16 which is our dependent variable

# So we decide to drop features A1,A4 and convert the DataFrame to a NumPy array
data = data.drop(['A1','A4'],axis=1)
test_data = test_data.drop(['A1','A4'],axis=1)

corr = data.corr()

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


data = data.values
test_data = test_data.values

# We will split independent variables(x) and dependent variable(y)
x = data[:,0:13] 
y = data[:,13]

#Scaling the data
scale = MinMaxScaler(feature_range=(0,1))
new_x = scale.fit_transform(x)
new_TD = scale.fit_transform(test_data)

# Split dataset into training set and test set
# 70% training and 30% test
x_train, x_test, y_train, y_test = train_test_split(new_x, y, test_size=0.30,random_state=109)

# Initialize a Logistic Regression classifier and fit model on train set
lmodel = LogisticRegression()
lmodel.fit(new_x,y)
predicted = lmodel.predict(x_test)

print("Accuracy: ", lmodel.score(x_test,y_test))
print(confusion_matrix(y_test,predicted))

# fpr,tpr,thresholds = roc_curve(y_test , predicted)
# f, ax = plt.subplots(figsize=(11, 9))
# plt.plot([0,1], [0,1], 'k--')
# plt.plot(fpr,tpr)
# plt.xlabel("False positive")
# plt.ylabel("True positive")
# plt.show()

# test_pred = lmodel.predict(new_TD)
# res = pd.DataFrame({ 'id' : range(1, test_pred.size+1 ,1)})
# res['Category'] = lmodel.inverse_transform(test_pred)
# res.to_csv("res8.csv",index=False)

# # Define the grid of values for tol and max_iter
# tol = [0.01, 0.001, 0.0001] 
# max_iter = [100, 150, 200]
# # Create a dictionary where tol and max_iter are keys and the lists of their values are corresponding values
# # Note: here tol is a key, max_iter is the corresponding value
# param_grid = dict(tol=tol, max_iter=max_iter)
# # Instantiate GridSearchCV with the required parameters
# grid_model = GridSearchCV(estimator=lmodel, param_grid=param_grid, cv=5)
# # Fit data to grid_model
# grid_model_result = grid_model.fit(new_x,y)
# # Summarize results
# best_score, best_params = grid_model_result.best_score_,grid_model_result.best_params_
# print("Best: %f using %s" % (best_score,best_params))

# lm = LogisticRegression(max_iter=100, tol=0.01)
# lm.fit(new_x,y)
# predicted = lm.predict(x_test)

# print("Accuracy: ", lm.score(x_test,y_test))
# print(confusion_matrix(y_test,predicted))

# rf = RandomForestClassifier(max_depth=15,min_samples_leaf=1,min_samples_split=2,n_estimators=500)
# rf.fit(x_train, y_train)
# y_pred = rf.predict(x_test)
# print("Random Forest classifier has accuracy of: ", rf.score(x_test, y_test))

# n_estimators = [100, 300, 500, 800, 1200]
# max_depth = [5, 8, 15, 25, 30]
# min_samples_split = [2, 5, 10, 15, 100]
# min_samples_leaf = [1, 2, 5, 10] 

# hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,  
#               min_samples_split = min_samples_split, 
#              min_samples_leaf = min_samples_leaf)

# gridF = GridSearchCV(rf, hyperF, cv = 3, verbose = 1, 
#                       n_jobs = -1)
# bestF = gridF.fit(new_x, y)

# bestF.best_params_
# print(bestF.best_params_)

