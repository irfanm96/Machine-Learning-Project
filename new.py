import numpy as np
import pandas as pd
from sklearn.impute import MissingIndicator
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


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


def encode_nominal_attributes(data_frame, columns):
    encoder = OneHotEncoder(dtype=int, sparse=True)

    nominal = pd.DataFrame(
        encoder.fit_transform(data_frame[columns])
            .toarray(), dtype=int)

    nominal.columns = encoder.get_feature_names().tolist()
    return nominal


def discretize_numerical_attributes(data_frame, columns):
    data_frame[columns].dtype = float
    return data_frame[columns]


def pre_processing(X):
    X.replace({'?': np.NaN}, inplace=True)
    # drop rows if all are missing values
    X.dropna(axis=0, thresh=1, inplace=True)

    # reset the index in data frame
    X.reset_index(inplace=True)
    X.drop(['index'], axis=1, inplace=True)

    nominal_columns = ['A1', 'A3', 'A4', 'A6', 'A8', 'A9', 'A11', 'A13']
    numerical_columns = ['A2', 'A5', 'A7', 'A10', 'A12', 'A14']

    x = impute_nominal_attributes(X, nominal_columns)
    y = impute_numerical_attributes(X, numerical_columns)

    k = merge_arrays_to_frames(x, nominal_columns, y, numerical_columns)

    no = encode_nominal_attributes(k, nominal_columns)
    nu = discretize_numerical_attributes(k, numerical_columns)
    return no
    # return pd.concat([no, nu], axis=1)


# replace 999 to NaN we can use this for replacing ? in dataset if needed
# X.replace({999.0: np.NaN}, inplace=True)

'''
We cannot use ordinal encoder since we dont know which are ordinal in those nominal attributes
'''
# read training dataset
data = pd.read_csv('data.csv', delimiter=',')
# get the labels
label = data['A16'].tolist()
# remove the labels from the data frame
data.drop(['A16'], axis=1, inplace=True)

# creating labelEncoder
le = preprocessing.LabelEncoder()

# set this as training data
X_train = data
y_train = le.fit_transform(label)

r = pre_processing(X_train)

# Create a Gaussian Classifier
model = GaussianNB()
model.fit(r.values, y_train)

# read training dataset
test_data = pd.read_excel('testdata.xlsx')
test_data.columns = data.columns

print(test_data)
# get the labels
test_label = test_data['A16'].tolist()
# remove the labels from the data frame
test_data.drop(['A16'], axis=1, inplace=True)

# set this as training data
X_test = test_data
y_test = le.fit_transform(test_label)

k = pre_processing(X_test)

# # Predict the response for test dataset
y_pred = model.predict(X_test.values)
#
# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
