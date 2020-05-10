import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

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

def pre_processing(df, type='train'):
    #Replacing non-standard data with NaN
    df.replace({'?': np.NaN}, inplace=True)
    #Drop rows if all are missing values
    df.dropna(axis=0, thresh=1, inplace=True)
    #Reset the index in data frame
    df.reset_index(inplace=True)
    df.drop(['index'], axis=1, inplace=True)

    #Inspect missing values in the dataset
    print(df.isna().sum())
    # Identify nominal and numeric attributes
    print(df.info())
    
    nominal_columns = ['A1', 'A3', 'A4', 'A6', 'A8', 'A9', 'A11', 'A13', 'A15']
    numerical_columns = ['A2', 'A5', 'A7', 'A10', 'A12', 'A14']

    x = impute_nominal_attributes(df, nominal_columns)
    y = impute_numerical_attributes(df, numerical_columns)
    k = merge_arrays_to_frames(x, nominal_columns, y, numerical_columns)
    
    #Check if any variables have any missing values after imputation
    print(pd.DataFrame(k).isna().sum())

    no = encode_nominal_attributes(k, nominal_columns, type=type)
    nu = discretize_numerical_attributes(k, numerical_columns)

    return pd.concat([no, nu], axis=1)


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

#Inspect data
print(data.head())
# print(test_data.head())
print("\n")

# Print summary statistics
print(data.describe())
# print(test_data.describe())
print("\n")

# Print DataFrame information
print(data.info())
# print(test_data.info())
print("\n")

#Inspect missing values in the dataset
print(data.tail(20))
# print(test_data.tail(20))
print("\n")

#Preprocessing Test and Training Datasets
data = pre_processing(data)
test_data = pre_processing(test_data, type='test')

#Inspect DataFrame after preprocessing
print(data.info())
print(data.head(20))
print(data.tail(20))
# print(test_data.info())
# print(test_data.head(20))
# print(test_data.tail(20))

# #Lets check if transformation has worked correctly
# print(data.head(20))

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
#70% training and 30% test
x_train, x_test, y_train, y_test = train_test_split(new_x, y, test_size=0.3,random_state=109)

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

#Initialize a K Neighbors Classifier and fit model on train set
kn = KNeighborsClassifier(n_neighbors=5)  
kn.fit(x_train, y_train)
kn_predicted = kn.predict(x_test)
#Get the accuracy score of the model
print("K Neighbors accuracy: ", kn.score(x_test, y_test))
#Confusion matrix of the model
print(confusion_matrix(y_test,kn_predicted))

#Initialize a Naive Bayes Classifier and fit model on train set
nb = GaussianNB()
nb.fit(x_train, y_train)
nb_predicted = nb.predict(x_test)
#Get the accuracy score of the model
print("Naive Bayes accuracy: ", nb.score(x_test, y_test))
#Confusion matrix of the model
print(confusion_matrix(y_test,nb_predicted))

#Predict test data with Logistic Regression model
lr_pred = lr.predict(new_TD)
lr_res = pd.DataFrame({ 'id' : range(1, lr_pred.size+1 ,1)})
lr_res['Category'] = encode.inverse_transform(lr_pred)
lr_res.to_csv("./output/lr_res.csv",index=False)

#Predict test data with Random Forest model
rf_pred = rf.predict(new_TD)
rf_res = pd.DataFrame({ 'id' : range(1, rf_pred.size+1 ,1)})
rf_res['Category'] = encode.inverse_transform(rf_pred)
rf_res.to_csv("./output/rf_res.csv",index=False)

#Predict test data with K Neighbors model
kn_pred = kn.predict(new_TD)
kn_res = pd.DataFrame({ 'id' : range(1, kn_pred.size+1 ,1)})
kn_res['Category'] = encode.inverse_transform(kn_pred)
kn_res.to_csv("./output/kn_res.csv",index=False)

#Predict test data with Naive Bayes model
nb_pred = nb.predict(new_TD)
nb_res = pd.DataFrame({ 'id' : range(1, nb_pred.size+1 ,1)})
nb_res['Category'] = encode.inverse_transform(nb_pred)
nb_res.to_csv("./output/nb_res.csv",index=False)