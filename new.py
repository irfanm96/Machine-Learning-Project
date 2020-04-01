import numpy as np
import pandas as pd
from sklearn.impute import MissingIndicator
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


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
    encoder = OneHotEncoder(dtype=np.int, sparse=True)

    nominal = pd.DataFrame(
        encoder.fit_transform(data_frame[columns])
            .toarray())

    nominal.columns = encoder.get_feature_names().tolist()
    return nominal


def discretize_numerical_attributes(data_frame, columns):
    disc = KBinsDiscretizer(n_bins=2, encode='onehot',
                            strategy='kmeans')

    numerical = pd.DataFrame(
        disc.fit_transform(data_frame[columns])
            .toarray())

    for i in range(len(disc.bin_edges_)):
        print("binary edge for ", columns[i], "numeric column")
        print(disc.bin_edges_[i])
        print()

    return numerical


def pre_processing(X):
    X.columns = ['sex', 'blood_type', 'edu_level', 'name']

    # drop rows if all are missing values
    X.dropna(axis=0, thresh=1, inplace=True)

    # reset the index in data frame
    X.reset_index(inplace=True)
    X.drop(['index'], axis=1, inplace=True)

    nominal_columns = ["sex", "blood_type"]
    numerical_columns = ['edu_level', 'name']

    x = impute_nominal_attributes(X, nominal_columns)
    y = impute_numerical_attributes(X, numerical_columns)

    X = merge_arrays_to_frames(x, nominal_columns, y, numerical_columns)

    no = encode_nominal_attributes(X, nominal_columns)

    nu = discretize_numerical_attributes(X, numerical_columns)

    return pd.concat([no, nu], axis=1)


# replace 999 to NaN we can use this for replacing ? in dataset if needed
# X.replace({999.0: np.NaN}, inplace=True)

'''
We cannot use ordinal encoder since we dont know which are ordinal in those nominal attributes
'''

from sklearn.preprocessing import OneHotEncoder

X = pd.DataFrame(
    [['M', 'O-', 70, 1],
     ['M', np.NaN, 90, 2],
     ['F', 'O+', 95, 3],
     ['F', 'O+', 96, 4],
     ['F', 'O+', 10, 4],
     ['M', 'B+', 96, 4],
     ['F', 'O+', 10, 4],
     ['F', 'B+', 96, 4],
     ['M', 'O+', 30, 4],
     ['F', 'B+', 96, 4],
     ['M', 'O+', 80, 4],
     ['F', 'AB', 35, 5],
     ['F', 'B+', np.NaN, 6]])

play = ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'No', 'Yes']

r = pre_processing(X)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(r.values, play, test_size=0.3)  # 70% training and 30% test

# Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets
model.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = model.predict(X_test)

# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
