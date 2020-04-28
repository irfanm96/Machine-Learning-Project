import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

def impute_nominal_attributes(data_frame, columns, strategy="most_frequent"):
    impurer = SimpleImputer(strategy=strategy)
    return impurer.fit_transform(data_frame[columns])

def impute_numerical_attributes(data_frame, columns, strategy="mean"):
    impute = SimpleImputer(missing_values=np.nan, strategy=strategy)
    return impute.fit_transform(data_frame[columns])

def merge_arrays_to_frames(array1, array1_columns, array2, array2_columns):
    a = pd.DataFrame(array1)
    a.columns = array1_columns

    b = pd.DataFrame(array2)
    b.columns = array2_columns

    return pd.concat([a, b], axis=1)

encoder = OneHotEncoder(dtype=int, sparse=True)

def encode_nominal_attributes(data_frame, columns, type='train'):
    if type == 'train':
        encoder.fit(data_frame[columns])

    nominal = pd.DataFrame(
        encoder.transform(data_frame[columns])
            .toarray())
    nominal.columns = encoder.get_feature_names().tolist()
    return nominal

def discretize_numerical_attributes(data_frame, columns):
    data_frame[columns].dtype = float
    return data_frame[columns]

def pre_processing(original, type='train'):
    original.replace({'?': np.NaN}, inplace=True)
    # drop rows if all are missing values
    original.dropna(axis=0, thresh=1, inplace=True)

    # reset the index in data frame
    original.reset_index(inplace=True)
    original.drop(['index'], axis=1, inplace=True)

    nominal_columns = ['A1', 'A3', 'A4', 'A6', 'A8', 'A9', 'A11', 'A13']
    numerical_columns = ['A2', 'A5', 'A7', 'A10', 'A12', 'A14']

    x = impute_nominal_attributes(original, nominal_columns)
    y = impute_numerical_attributes(original, numerical_columns)

    k = merge_arrays_to_frames(x, nominal_columns, y, numerical_columns)

    no = encode_nominal_attributes(k, nominal_columns, type=type)
    nu = discretize_numerical_attributes(k, numerical_columns)
    return pd.concat([no, nu], axis=1)

'''
We cannot use ordinal encoder since we dont know which are ordinal in those nominal attributes
'''

# read training dataset
data = pd.read_csv('./input/data.csv', delimiter=',')

# get the labels
label = data['A16'].tolist()
# remove the labels from the data frame
data.drop(['A16'], axis=1, inplace=True)

# creating labelEncoder
le = preprocessing.LabelEncoder()

r = pre_processing(data)
target = le.fit_transform(label)

# Split dataset into training set and test set
# 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(r.values, target, test_size=0.3,random_state=109)  

# read training dataset
test_data = pd.read_csv('./input/testdata.csv', delimiter=',')

original_data = test_data.copy()

b = pre_processing(test_data, type='test')

# Scaling X_train, X_test, fullData, given TestData
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX_train = scaler.fit_transform(X_train)
rescaledX_test = scaler.transform(X_test)
rescaledX_Data = scaler.fit_transform(r.values)
rescaledX_testData = scaler.transform(b.values)

# Fitting logistic regression with default parameter values
logreg = LogisticRegression()
logreg.fit(rescaledX_train, y_train)

# Using the trained model to predict instances from the test set
y_pred = logreg.predict(rescaledX_test)

# Getting the accuracy score of predictive model
print("Logistic regression classifier has accuracy of: ", logreg.score(rescaledX_test, y_test))

# test_pred = logreg.predict(rescaledX_testData)

# print()
# print("The predicted results for the test data")
# print(le.inverse_transform(test_pred))

# original_data['A16'] = le.inverse_transform(test_pred)
# # original_data.to_csv("predictions2.csv",index=False)

# res = pd.DataFrame({ 'id' : range(1, test_pred.size+1 ,1)})
# res['Category'] = le.inverse_transform(test_pred)
# res.to_csv("res2.csv",index=False)

# from sklearn.model_selection import GridSearchCV

# # Define the grid of values for tol and max_iter
# tol = [0.01,0.001,0.0001]
# max_iter = [100,150,200]

# # Create a dictionary where tol and max_iter are keys and the lists of their values are corresponding values
# # Note: here tol is a key, max_iter is the corresponding value
# param_grid = dict(tol=tol, max_iter=max_iter)

# # Instantiate GridSearchCV with the required parameters
# grid_model = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5)

# # Use scaler to rescale X and assign it to rescaledX

# # Fit data to grid_model
# grid_model_result = grid_model.fit(rescaledX_Data, target)

# # Summarize results
# best_score, best_params = grid_model_result.best_score_,grid_model_result.best_params_
# print("Best: %f using %s" % (best_score,best_params))

lr = LogisticRegression(max_iter=100,tol=.01)
lr.fit(rescaledX_Data,target)

# Using the trained model to predict instances from the test set
y_pred = lr.predict(rescaledX_test)

# Getting the accuracy score of predictive model
print("Logistic regression classifier has accuracy of: ", lr.score(rescaledX_test, y_test))

test_pred = lr.predict(rescaledX_testData)

# print()
# print("The predicted results for the test data")
# print(le.inverse_transform(test_pred))

original_data['A16'] = le.inverse_transform(test_pred)
# # original_data.to_csv("predictions2.csv",index=False)

res = pd.DataFrame({ 'id' : range(1, test_pred.size+1 ,1)})
res['Category'] = le.inverse_transform(test_pred)
res.to_csv("res5.csv",index=False)