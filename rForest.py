import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

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

    nominal_columns = ['A3','A6', 'A8', 'A11', 'A13']
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

#Read training & testing dataset
data = pd.read_csv('./input/data.csv', delimiter=',')
test_data = pd.read_csv('./input/testdata.csv', delimiter=',')

# get the labels
label = data['A16'].tolist()
# remove the labels from the data frame
data.drop(['A16'], axis=1, inplace=True)

data = data.drop(['A1','A4','A9'],axis=1)
test_data = test_data.drop(['A1','A4','A9'],axis=1)

# creating labelEncoder
le = preprocessing.LabelEncoder()

r = pre_processing(data)
target = le.fit_transform(label)
b = pre_processing(test_data, type='test')

# Split dataset into training set and test set
# 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(r.values, target, test_size=0.3,random_state=109)

# Scaling X_train and X_test
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX_Data = scaler.fit_transform(r.values)
rescaledX_train = scaler.fit_transform(X_train)
rescaledX_test = scaler.transform(X_test)
rescaledX_tData = scaler.fit_transform(b.values)

rf = RandomForestClassifier(n_estimators=500)
rf.fit(rescaledX_Data, target)
y_pred = rf.predict(rescaledX_test)
print("Random Forest classifier has accuracy of: ", rf.score(rescaledX_test, y_test))

lmodel = LogisticRegression()
lmodel.fit(rescaledX_Data, target)
predicted = lmodel.predict(rescaledX_test)
print("Logistic Regression Accuracy: ", lmodel.score(rescaledX_test,y_test))
print(confusion_matrix(y_test,predicted))

rf_pred = rf.predict(rescaledX_tData)
lr_pred = lmodel.predict(rescaledX_tData)

res = pd.DataFrame({ 'id' : range(1, rf_pred.size+1 ,1)})
res['rf'] = le.inverse_transform(rf_pred)
res['lr'] = le.inverse_transform(lr_pred)
res.to_csv("./output/rflr.csv",index=False)