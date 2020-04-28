import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier  
import matplotlib.pyplot as plt
import seaborn as sns

#Read Training dataset
data = pd.read_csv('./input/data.csv', delimiter=',')
test_data = pd.read_csv('./input/testdata.csv', delimiter=',')

# print(data.head())
#Replacing non-standard data with NaN
data = data.replace( "?" , np.NaN)
test_data = test_data.replace( "?" , np.NaN)

#Replacing NaN values in numeric variables with mean values for that variable
data.fillna(data.mean(), inplace=True)
test_data.fillna(data.mean(), inplace=True)

# print(data.isnull().values.sum())
# print(test_data.isnull().values.sum())

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
# print(data.info())
# print(test_data.info())

#Creating correlation matrix for our data to check correlation of variables and possibility of multicollinearity
corr = data.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(15, 10),sharex=True, sharey=True)

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, annot=True,annot_kws={"size": 7}, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .8})

# plt.show()

#We can observe feature A4,A3 highly correlated, this might lead to multicollinearity
#we can also observe that feature A1,A9 have very less correlation with variable A16 which is our dependent variable

# # So we decide to drop features A1,A4,A9 and convert the DataFrame to a NumPy array
data = data.drop(['A1','A4','A9'],axis=1)
test_data = test_data.drop(['A1','A4','A9'],axis=1)

# corr = data.corr()

# mask = np.zeros_like(corr, dtype=np.bool)
# mask[np.triu_indices_from(mask)] = True
# # Set up the matplotlib figure
# f, ax = plt.subplots(figsize=(11, 9))
# # Generate a custom diverging colormap
# cmap = sns.diverging_palette(220, 10, as_cmap=True)
# # Draw the heatmap with the mask and correct aspect ratio
# sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
#             square=True, linewidths=.5, cbar_kws={"shrink": .5})

# # plt.show()

data = data.values
test_data = test_data.values

# # We will split independent variables(x) and dependent variable(y)
x = data[:,0:12] 
y = data[:,12]

#Scaling the data
scale = MinMaxScaler(feature_range=(0,1))
new_x = scale.fit_transform(x)
new_TD = scale.fit_transform(test_data)

# Split dataset into training set and test set
# 70% training and 30% test
x_train, x_test, y_train, y_test = train_test_split(new_x, y, test_size=0.30,random_state=109)

# Initialize a Logistic Regression classifier and fit model on train set
lmodel = LogisticRegression()
lmodel.fit(x_train,y_train)
predicted = lmodel.predict(x_test)
print("Logistic Regression Accuracy: ", lmodel.score(x_test,y_test))
print(confusion_matrix(y_test,predicted))

# rf = RandomForestClassifier(n_estimators=500)
# rf.fit(x_train, y_train)
# y_pred = rf.predict(x_test)
# print("Random Forest accuracy: ", rf.score(x_test, y_test))
# print(confusion_matrix(y_test,y_pred))