import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def impute_nominal_attributes(data_frame, columns, strategy="constant"):
    impurer = SimpleImputer(strategy=strategy)
    return impurer.fit_transform(data_frame[columns])

def impute_numerical_attributes(data_frame, columns, strategy="median"):
    impute = SimpleImputer(missing_values=np.nan, strategy=strategy)
    return impute.fit_transform(data_frame[columns])

def merge_arrays_to_frames(array1, array1_columns, array2, array2_columns):
    a = pd.DataFrame(array1)
    a.columns = array1_columns

    b = pd.DataFrame(array2)
    b.columns = array2_columns

    return pd.concat([a, b], axis=1)

def encode_nominal_attributes(df):
    df["A1"]=df["A1"].replace('a',1,regex=True)
    df["A1"]=df["A1"].replace('b',2,regex=True)

    df["A3"]=df["A3"].replace('l',1,regex=True)
    df["A3"]=df["A3"].replace('u',2,regex=True)
    df["A3"]=df["A3"].replace('y',3,regex=True)

    df["A4"]=df["A4"].replace('gg',1,regex=True)
    df["A4"]=df["A4"].replace('g',2,regex=True)
    df["A4"]=df["A4"].replace('p',3,regex=True)

    df["A6"]=df["A6"].replace('aa',1,regex=True)
    df["A6"]=df["A6"].replace('cc',2,regex=True)
    df["A6"]=df["A6"].replace('ff',3,regex=True)
    df["A6"]=df["A6"].replace('m',4,regex=True)
    df["A6"]=df["A6"].replace('k',5,regex=True)
    df["A6"]=df["A6"].replace('j',6,regex=True)
    df["A6"]=df["A6"].replace('r',7,regex=True)
    df["A6"]=df["A6"].replace('w',8,regex=True)
    df["A6"]=df["A6"].replace('q',9,regex=True)
    df["A6"]=df["A6"].replace('c',10,regex=True)
    df["A6"]=df["A6"].replace('x',11,regex=True)
    df["A6"]=df["A6"].replace('i',12,regex=True)
    df["A6"]=df["A6"].replace('d',13,regex=True)
    df["A6"]=df["A6"].replace('e',14,regex=True)

    df["A8"]=df["A8"].replace(True,1,regex=True)
    df["A8"]=df["A8"].replace(False,0,regex=True)

    df["A9"]=df["A9"].replace('v',1,regex=True)
    df["A9"]=df["A9"].replace('h',2,regex=True)
    df["A9"]=df["A9"].replace('bb',3,regex=True)
    df["A9"]=df["A9"].replace('ff',4,regex=True)
    df["A9"]=df["A9"].replace('j',5,regex=True)
    df["A9"]=df["A9"].replace('z',6,regex=True)
    df["A9"]=df["A9"].replace('o',7,regex=True)
    df["A9"]=df["A9"].replace('dd',8,regex=True)
    df["A9"]=df["A9"].replace('n',9,regex=True)

    df["A11"]=df["A11"].replace(True,1,regex=True)
    df["A11"]=df["A11"].replace(False,0,regex=True)

    df["A13"]=df["A13"].replace(True,1,regex=True)
    df["A13"]=df["A13"].replace(False,0,regex=True)

    df["A15"]=df["A15"].replace('g',1,regex=True)
    df["A15"]=df["A15"].replace('s',2,regex=True)
    df["A15"]=df["A15"].replace('p',3,regex=True)
    
    return df

def pre_processing(original, type='train'):
    #Replacing non-standard data with NaN
    original.replace({'?': np.NaN}, inplace=True)
    #Drop rows if all are missing values
    original.dropna(axis=0, thresh=1, inplace=True)
    #Reset the index in data frame
    original.reset_index(inplace=True)
    original.drop(['index'], axis=1, inplace=True)

    nominal_columns = ['A1', 'A3', 'A4', 'A6', 'A8', 'A9', 'A11', 'A13', 'A15']
    numerical_columns = ['A2', 'A5', 'A7', 'A10', 'A12', 'A14']

    x = impute_nominal_attributes(original, nominal_columns)
    y = impute_numerical_attributes(original, numerical_columns)

    k = merge_arrays_to_frames(x, nominal_columns, y, numerical_columns)

    df = encode_nominal_attributes(pd.DataFrame(k))

    return df


#Read Training dataset
data = pd.read_csv('./input/data.csv', delimiter=',')
test_data = pd.read_csv('./input/testdata.csv', delimiter=',')

#Get the labels
label = data['A16'].tolist()
#Remove the labels from the data frame
data = data.drop(['A16'],axis=1)
#Converting non-numeric label feature into numeric data using LabelEncoder
encode = preprocessing.LabelEncoder()
label = encode.fit_transform(label)

#Preprocessing Test and Training Datasets
data = pre_processing(data)
test_data = pre_processing( test_data, type='test' )

#To analyze add label to Data
data['A16'] = label

#Creating correlation matrix for our data to check correlation of variables and possibility of multicollinearity
corr = data.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

#Set up the matplotlib figure
f, ax = plt.subplots(figsize=(12, 10),sharex=True, sharey=True)

#Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

#Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, annot=True,annot_kws={"size": 7}, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .8})

#Feature A4,A3 highly correlated, this might lead to multicollinearity
#Feature A1 has very less correlation with variable A16 which is our dependent variable

#So we decide to drop features A1,A4
data = data.drop(['A4','A1'],axis=1)
test_data = test_data.drop(['A4','A1'],axis=1)

corr = data.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
#Set up the matplotlib figure
f, ax = plt.subplots(figsize=(12, 10),sharex=True, sharey=True)
#Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, annot=True,annot_kws={"size": 7}, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .8})

# plt.show()

#Remove the labels from the data frame
data = data.drop(['A16'],axis=1)

#Convert the DataFrame to a NumPy array
data = data.values
test_data = test_data.values

#Split independent variables x and dependent variable y
x = data[:,0:]
y = label

#Scaling the data using MinMaxScaler
scale = MinMaxScaler(feature_range=(0,1))
new_x = scale.fit_transform(x)
new_TD = scale.fit_transform(test_data)

#Split dataset into training set and test set
#80% training and 20% test
x_train, x_test, y_train, y_test = train_test_split(new_x, y, test_size=0.2,random_state=109)

#Initialize a Logistic Regression Classifier and fit model on train set
lr = LogisticRegression()
lr.fit(x_train, y_train)
lr_predicted = lr.predict(x_test)
#Get the accuracy score of the model
print("Logistic Regression Accuracy: ", lr.score(x_test,y_test))
#Confusion matrix of the model
print(confusion_matrix(y_test,lr_predicted))

#Initialize a Random Forest Classifier and fit model on train set
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
rf_predicted = rf.predict(x_test)
#Get the accuracy score of the model
print("Random Forest accuracy: ", rf.score(x_test, y_test))
#Confusion matrix of the model
print(confusion_matrix(y_test,rf_predicted))

# #Predict test data with Logistic Regression model
# lr_pred = lr.predict(new_TD)
# lr_res = pd.DataFrame({ 'id' : range(1, lr_pred.size+1 ,1)})
# lr_res['Category'] = encode.inverse_transform(lr_pred)
# lr_res.to_csv("./output/lr_Result.csv",index=False)

# #Predict test data with Random Forest model
# rf_pred = rf.predict(new_TD)
# rf_res = pd.DataFrame({ 'id' : range(1, rf_pred.size+1 ,1)})
# rf_res['Category'] = encode.inverse_transform(rf_pred)
# rf_res.to_csv("./output/rf_Result.csv",index=False)

# #Grid searching and finding Best model parameters

# # Define the grid of values for n_estimators, max_depth, min_samples_split, min_samples_leaf
# n_estimators = [100, 300, 500, 800, 1200, 1500]
# max_depth = [5, 8, 15, 25, 30]
# min_samples_split = [2, 5, 10, 15, 100]
# min_samples_leaf = [1, 2, 5, 10] 
# #Create a dictionary
# rf_param_grid = dict(n_estimators = n_estimators, max_depth = max_depth,
#                 min_samples_split = min_samples_split, 
#                 min_samples_leaf = min_samples_leaf)
# #Instantiate GridSearchCV with the required parameters
# rf_grid_model = GridSearchCV(estimator=rf, param_grid=rf_param_grid, cv = 3, verbose = 1, n_jobs = -1)
# #Fit data to grid_model
# rf_grid_model_result = rf_grid_model.fit(x_train, y_train)

# #Grid model results
# rf_best_score, rf_best_params = rf_grid_model_result.best_score_,rf_grid_model_result.best_params_
# print("Best: %f using %s" % (rf_best_score,rf_best_params))

# #Define the grid of values for tol and max_iter
# tol = [0.01,0.001,0.0001]
# max_iter = [100,150,200]
# #Create a dictionary
# lr_param_grid = dict(tol=tol, max_iter=max_iter)
# #Instantiate GridSearchCV with the required parameters
# lr_grid_model = GridSearchCV(estimator=lr, param_grid=lr_param_grid, cv=5)
# #Fit data to grid_model
# lr_grid_model_result = lr_grid_model.fit(x_train, y_train)

# #Grid model results
# lr_best_score, lr_best_params = lr_grid_model_result.best_score_,lr_grid_model_result.best_params_
# print("Best: %f using %s" % (lr_best_score, lr_best_score))

#Initialize a Logistic Regression Classifier with best fit model parameters and fit model on train set
improved_lr = LogisticRegression(max_iter=100, tol=0.01)
improved_lr.fit(x_train, y_train)
#Predict test data with Logistic Regression model
lr_pred = improved_lr.predict(new_TD)
lr_res = pd.DataFrame({ 'id' : range(1, lr_pred.size+1 ,1)})
lr_res['Category'] = encode.inverse_transform(lr_pred)
lr_res.to_csv("./output/Results.csv",index=False)

#Initialize a Random Forest Classifier with best fit model parameters and fit model on train set
improved_rf = RandomForestClassifier(n_estimators=500)
improved_rf.fit(x_train, y_train)
#Predict test data with Random Forest model
rf_pred = improved_rf.predict(new_TD)
rf_res = pd.DataFrame({ 'id' : range(1, rf_pred.size+1 ,1)})
rf_res['Category'] = encode.inverse_transform(rf_pred)
rf_res.to_csv("./output/improved_rf_res.csv",index=False)