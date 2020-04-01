import numpy as np
import pandas as pd
from sklearn.impute import MissingIndicator
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing


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
    disc = KBinsDiscretizer(n_bins=3, encode='onehot',
                            strategy='uniform')

    numerical = pd.DataFrame(
        disc.fit_transform(data_frame[columns])
            .toarray())

    print(disc.bin_edges_)

    return numerical


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
     ['F', 'AB', 35, 5],
     ['F', 'B+', np.NaN, 6]])

play = ['No', 'No', 'Yes', 'Yes', 'Yes', 'No']

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

r = pd.concat([no, nu], axis=1)

print(r)

# creating labelEncoder
le = preprocessing.LabelEncoder()

label = le.fit_transform(play)

# Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets
model.fit(r.values, label)

# Predict Output
predicted = model.predict([[0, 1, 0, 0, 0, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]])  # 0:Overcast, 2:Mild
print("Predicted Value:", predicted)
