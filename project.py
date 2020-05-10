import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
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

def impute_numerical_attributes(data_frame, columns, strategy="mean"):
    impute = SimpleImputer(missing_values=np.nan, strategy=strategy)
    return impute.fit_transform(data_frame[columns])

def merge_arrays_to_frames(array1, array1_columns, array2, array2_columns):
    a = pd.DataFrame(array1)
    a.columns = array1_columns

    b = pd.DataFrame(array2)
    b.columns = array2_columns

    return pd.concat([a, b], axis=1)

def encode_nominal_attributes(df,nominal_columns):
    # Identify unique values in each nominal attributes
    for i in nominal_columns:
        print(i,np.sort(df[i].unique()))
    # Unique Values
    # A1 ['a' 'b']
    # A3 ['l' 'u' 'y']
    # A4 ['g' 'gg' 'p']
    # A6 ['aa' 'c' 'cc' 'd' 'e' 'ff' 'i' 'j' 'k' 'm' 'q' 'r' 'w' 'x']
    # A8 [False True]
    # A9 ['bb' 'dd' 'ff' 'h' 'j' 'n' 'o' 'v' 'z']
    # A11 [False True]
    # A13 [False True]
    # A15 ['g' 'p' 's']

    # Create dictionary with unique values 
    nominal_attributes_values = { 'A1': ['a', 'b'],
        'A3': ['l', 'u', 'y'],
        'A4': ['gg', 'g', 'p'],
        'A6': ['aa','cc', 'ff', 'c', 'd', 'e', 'i', 'j', 'k', 'm', 'q', 'r', 'w', 'x'],
        'A8': [False, True],
        'A9': ['bb', 'ff', 'dd', 'h', 'j', 'n', 'o', 'v', 'z'],
        'A11': [False, True],
        'A13': [False, True],
        'A15': ['g', 'p', 's'] }

    # Replace each nominal attributes value with a unique numerica value 
    for key, values in nominal_attributes_values.items():
        for i,value in enumerate(values): 
            df[key] = df[key].replace(value,i,regex=True)

    return df

def pre_processing(df, type='train'):
    # Replacing non-standard data with NaN
    df.replace({'?': np.NaN}, inplace=True)
    # Drop rows if all are missing values
    df.dropna(axis=0, thresh=1, inplace=True)
    # Reset the index in data frame
    df.reset_index(inplace=True)
    df.drop(['index'], axis=1, inplace=True)

    # # Identify nominal and numeric attributes
    # print(df.info())
    nominal_columns = ['A1', 'A3', 'A4', 'A6', 'A8', 'A9', 'A11', 'A13', 'A15']
    numerical_columns = ['A2', 'A5', 'A7', 'A10', 'A12', 'A14']
    
    x = impute_nominal_attributes(df, nominal_columns)
    y = impute_numerical_attributes(df, numerical_columns)
    k = merge_arrays_to_frames(x, nominal_columns, y, numerical_columns)
    
    df = encode_nominal_attributes( pd.DataFrame(k), nominal_columns )
    # # Inspect data after pre processing
    # print(df.dtypes)
    # print(df.head(20))
    # print(df.head(20))

    # # Inspect numerical attributes value ranges 
    # for i in numerical_columns:
    #     print(i,np.sort(df[i].unique()))

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

# Analys    e Class Distribution
unique, sizes = np.unique(label,return_counts=True)
print(sizes)
print(unique)
labels = 'Faliure', 'Success'
colors = ['yellowgreen', 'lightskyblue']
# Plot pie chart
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
# # Show Plot
# plt.show()

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

# # Show Plot
# plt.show()

#Feature A4,A3 highly correlated, this might lead to multicollinearity
#Features A1,A9 have very less correlation with variable A16 which is our dependent variable

#So we decide to drop features A1,A4
data = data.drop(['A4','A1','A9'],axis=1)
test_data = test_data.drop(['A4','A1','A9'],axis=1)

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

# # Show Plot
# plt.show()

#Remove the labels from the data frame
data = data.drop(['A16'],axis=1)

#Convert the DataFrame to a NumPy array
data = data.values
test_data = test_data.values

#Split independent variables x and dependent variable y
x = data[:,0:]
y = label

#Split dataset into training set and test set
#80% training and 20% test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=109)

#Scaling the data using MinMaxScaler
scale = MinMaxScaler(feature_range=(0,1))
x_train = scale.fit_transform(x_train)
x_test = scale.fit_transform(x_test)
new_TD = scale.fit_transform(test_data)

#Initialize a Logistic Regression Classifier and fit model on train set
lr = LogisticRegression()
lr.fit(x_train, y_train)
lr_predicted = lr.predict(x_test)
#Get the accuracy score of the model
print("Logistic Regression training accuracy: ", lr.score(x_train, y_train))
print("Logistic Regression testing accuracy: ", lr.score(x_test,y_test))
#Confusion matrix of the model
print(confusion_matrix(y_test,lr_predicted))

#Initialize a Random Forest Classifier and fit model on train set
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
rf_predicted = rf.predict(x_test)
#Get the accuracy score of the model
print("Random Forest training accuracy: ", rf.score(x_train, y_train))
print("Random Forest testing accuracy: ", rf.score(x_test, y_test))
#Confusion matrix of the model
print(confusion_matrix(y_test,rf_predicted))

# #Predict test data with Logistic Regression model
# lr_pred = lr.predict(new_TD)
# lr_res = pd.DataFrame({ 'id' : range(1, lr_pred.size+1 ,1)})
# lr_res['Category'] = encode.inverse_transform(lr_pred)
# lr_res.to_csv("./output/lr_Result.csv",index=False)

#Predict test data with Random Forest model
rf_pred = rf.predict(new_TD)
rf_res = pd.DataFrame({ 'id' : range(1, rf_pred.size+1 ,1)})
rf_res['Category'] = encode.inverse_transform(rf_pred)
rf_res.to_csv("./output/rf_Result.csv",index=False)

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

#Define the grid of values for tol and max_iter
tol = [0.01,0.001,0.0001]
max_iter = [100,150,200]
C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
#Create a dictionary
lr_param_grid = dict(tol=tol, max_iter=max_iter, C=C)
#Instantiate GridSearchCV with the required parameters
lr_grid_model = GridSearchCV(estimator=lr, param_grid=lr_param_grid, cv=5)
#Fit data to grid_model
lr_grid_model_result = lr_grid_model.fit(x_train, y_train)

#Grid model results
lr_best_score, lr_best_params = lr_grid_model_result.best_score_,lr_grid_model_result.best_params_
print("Best: %f using %s" % (lr_best_score, lr_best_params))

#Initialize a Logistic Regression Classifier with best fit model parameters and fit model on train set
improved_lr = LogisticRegression(max_iter=100, tol=0.01, C=0.1)
improved_lr.fit(x_train, y_train)
#Get the accuracy score of the model
print("Random Forest testing accuracy: ", improved_lr.score(x_train, y_train))
print("Random Forest accuracy: ", improved_lr.score(x_test, y_test))
#Predict test data with Logistic Regression model
lr_pred = improved_lr.predict(new_TD)
lr_res = pd.DataFrame({ 'id' : range(1, lr_pred.size+1 ,1)})
lr_res['Category'] = encode.inverse_transform(lr_pred)
lr_res.to_csv("./output/Results.csv",index=False)

# #Initialize a Random Forest Classifier with best fit model parameters and fit model on train set
# improved_rf = RandomForestClassifier(n_estimators=500)
# improved_rf.fit(x_train, y_train)
# #Predict test data with Random Forest model
# rf_pred = improved_rf.predict(new_TD)
# rf_res = pd.DataFrame({ 'id' : range(1, rf_pred.size+1 ,1)})
# rf_res['Category'] = encode.inverse_transform(rf_pred)
# rf_res.to_csv("./output/improved_rf_res.csv",index=False)